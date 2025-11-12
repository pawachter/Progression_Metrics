"""
Unified Callbacks Module
All callbacks and factory for CNN training with domain adaptation
"""

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import logging
import yaml
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
from distance_metrics import (
    compute_mmd,
    compute_kl_divergence,
    compute_wasserstein_distance,
    compute_entropy
)


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def get_logger(name, log_dir='logs', log_file='training.log'):
    """Create/get shared logger to avoid conflicts"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, log_file))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        logger.addHandler(fh)
        logger.propagate = False  # Prevent propagation to root logger (console)
    return logger


def log_and_flush(logger, message):
    """Log message and immediately flush to disk for real-time logging"""
    logger.info(message)
    for handler in logger.handlers:
        handler.flush()


# ============================================================================
# TRAINING METRICS LOGGING CALLBACK
# ============================================================================

class TrainingMetricsLogger(tf.keras.callbacks.Callback):
    """Log training and validation metrics (loss, accuracy) to training.log"""
    
    def __init__(self, log_dir='logs'):
        super(TrainingMetricsLogger, self).__init__()
        self.log_dir = log_dir
        self.logger = get_logger('TrainingMetrics', log_dir, 'training.log')
    
    def on_train_begin(self, logs=None):
        log_and_flush(self.logger, "=" * 60)
        log_and_flush(self.logger, "Training Started")
        log_and_flush(self.logger, "=" * 60)
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        # Extract metrics from logs
        train_loss = logs.get('loss', 'N/A')
        train_acc = logs.get('accuracy', 'N/A')
        val_loss = logs.get('val_loss', 'N/A')
        val_acc = logs.get('val_accuracy', 'N/A')
        
        # Format the log message
        message = f"Epoch {epoch + 1}: "
        message += f"Train Loss = {train_loss:.6f}, " if isinstance(train_loss, (int, float)) else f"Train Loss = {train_loss}, "
        message += f"Train Acc = {train_acc:.6f}, " if isinstance(train_acc, (int, float)) else f"Train Acc = {train_acc}, "
        message += f"Val Loss = {val_loss:.6f}, " if isinstance(val_loss, (int, float)) else f"Val Loss = {val_loss}, "
        message += f"Val Acc = {val_acc:.6f}" if isinstance(val_acc, (int, float)) else f"Val Acc = {val_acc}"
        
        log_and_flush(self.logger, message)
    
    def on_train_end(self, logs=None):
        log_and_flush(self.logger, "=" * 60)
        log_and_flush(self.logger, "Training Completed")
        log_and_flush(self.logger, "=" * 60)


# ============================================================================
# CONVERGENCE MONITORING CALLBACKS
# ============================================================================

class GradientAnalysisCallback(tf.keras.callbacks.Callback):
    """Analyze gradient statistics: norm, SNR, cosine similarity - OPTIMIZED WITH ASYNC"""
    
    def __init__(self, log_dir='logs', K=50, ema_alpha=0.99, epsilon=1e-8, max_workers=2):
        super(GradientAnalysisCallback, self).__init__()
        self.log_dir = log_dir
        self.K = K
        self.ema_alpha = tf.constant(ema_alpha, dtype=tf.float32)
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)
        # Use deque for efficient pop/append and store as tensors
        self.gradients_history = deque(maxlen=K)
        self.ema_cosine_similarity = None
        self.logger = get_logger('GradientAnalysis', log_dir, 'training.log')
        
        # Threading support
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.pending_futures = []

    def on_train_begin(self, logs=None):
        self.gradients_history.clear()
        self.ema_cosine_similarity = None
        self.pending_futures = []

    def on_batch_end(self, batch, logs=None):
        if not hasattr(self.model, 'current_gradients') or self.model.current_gradients is None:
            return
        
        # Keep as TensorFlow tensor - NO numpy conversion
        gradients = self.flatten_gradients(self.model.current_gradients)
        global_gradient_norm = tf.norm(gradients, ord=2)
        
        # Log gradient norm immediately (fast operation)
        log_and_flush(self.logger, f"Batch {batch + 1}: Global Gradient Norm = {global_gradient_norm.numpy():.6f}")
        
        # Store as numpy for thread safety (TensorFlow operations in threads can be problematic)
        gradients_numpy = gradients.numpy()
        
        # Thread-safe update of history
        with self.lock:
            self.gradients_history.append(gradients_numpy)
            history_snapshot = list(self.gradients_history)
            ema_snapshot = self.ema_cosine_similarity
        
        # Submit async computation
        future = self.executor.submit(
            self._compute_metrics_async, 
            batch, 
            history_snapshot, 
            ema_snapshot,
            self.K,
            float(self.ema_alpha.numpy()),
            float(self.epsilon.numpy())
        )
        self.pending_futures.append(future)
        
        # Clean up completed futures
        self.pending_futures = [f for f in self.pending_futures if not f.done()]

    def _compute_metrics_async(self, batch, history_snapshot, ema_snapshot, K, ema_alpha, epsilon):
        """Compute SNR and cosine similarity asynchronously in a separate thread"""
        try:
            # Compute SNR when we have K gradients
            if len(history_snapshot) == K:
                snr = self._compute_snr_numpy(history_snapshot, epsilon)
                log_and_flush(self.logger, f"Batch {batch + 1}: SNR = {snr:.6f}")
            
            # Compute cosine similarity
            if len(history_snapshot) > 1:
                cosine_similarity = self._compute_cosine_similarity_numpy(
                    history_snapshot[-1], 
                    history_snapshot[-2],
                    epsilon
                )
                
                if ema_snapshot is None:
                    new_ema = cosine_similarity
                else:
                    new_ema = ema_alpha * ema_snapshot + (1.0 - ema_alpha) * cosine_similarity
                
                # Thread-safe update of EMA
                with self.lock:
                    self.ema_cosine_similarity = new_ema
                
                log_and_flush(self.logger, f"Batch {batch + 1}: EMA Cosine Similarity = {new_ema:.6f}")
        except Exception as e:
            self.logger.error(f"Error in async metric computation: {e}")

    def _compute_snr_numpy(self, gradients_history, epsilon):
        """Compute SNR using numpy operations"""
        gradients_stack = np.stack(gradients_history, axis=0)
        mean_gradient = np.mean(gradients_stack, axis=0)
        variance_gradient = np.mean(np.square(gradients_stack - mean_gradient), axis=0)
        snr = np.linalg.norm(mean_gradient) / np.sqrt(np.sum(variance_gradient) + epsilon)
        return snr

    def _compute_cosine_similarity_numpy(self, grad1, grad2, epsilon):
        """Compute cosine similarity using numpy operations"""
        dot_product = np.sum(grad1 * grad2)
        norm_grad1 = np.linalg.norm(grad1)
        norm_grad2 = np.linalg.norm(grad2)
        cosine_similarity = dot_product / (norm_grad1 * norm_grad2 + epsilon)
        return cosine_similarity

    def on_train_end(self, logs=None):
        """Wait for all pending computations and cleanup"""
        log_and_flush(self.logger, "Waiting for pending gradient analysis computations...")
        for future in self.pending_futures:
            try:
                future.result(timeout=30)  # Wait max 30 seconds per task
            except Exception as e:
                self.logger.error(f"Error waiting for async task: {e}")
        self.executor.shutdown(wait=True)
        log_and_flush(self.logger, "Gradient analysis cleanup complete")

    # Remove @tf.function - this needs to run in eager mode
    def flatten_gradients(self, gradients):
        """Flatten gradients using pure TensorFlow ops"""
        flattened = []
        for grad in gradients:
            if grad is not None:
                flattened.append(tf.reshape(grad, [-1]))
        return tf.concat(flattened, axis=0)


class FisherTraceCallback(tf.keras.callbacks.Callback):
    """Compute Fisher Information trace on validation sets - OPTIMIZED WITH ASYNC"""
    
    def __init__(self, val_dataset_a, n_steps=100, log_dir='logs', num_batches=10, batch_size=32, max_workers=1):
        super(FisherTraceCallback, self).__init__()
        # Pre-batch, cache, repeat, and prefetch the dataset for optimal performance
        self.val_dataset_a = (val_dataset_a
                             .batch(batch_size)
                             .take(num_batches)
                             .cache()
                             .repeat()
                             .prefetch(tf.data.AUTOTUNE))
        self.n_steps = n_steps
        self.log_dir = log_dir
        self.step_counter = 0
        self.num_batches = num_batches
        self.logger = get_logger('FisherTrace', log_dir, 'training.log')
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        # Threading support
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures = []

    def on_train_begin(self, logs=None):
        self.step_counter = 0
        self.pending_futures = []

    def on_batch_end(self, batch, logs=None):
        self.step_counter += 1
        if self.step_counter % self.n_steps == 0:
            # Submit async computation
            future = self.executor.submit(
                self._compute_fisher_async,
                self.step_counter
            )
            self.pending_futures.append(future)
            
            # Clean up completed futures
            self.pending_futures = [f for f in self.pending_futures if not f.done()]

    def _compute_fisher_async(self, step_counter):
        """Compute Fisher trace asynchronously in a separate thread"""
        try:
            fisher_trace_a = self.compute_fisher_trace_tf()
            log_and_flush(self.logger, f"Step {step_counter}: Fisher Trace (Val Set A) = {fisher_trace_a.numpy():.6f}")
        except Exception as e:
            self.logger.error(f"Error in async Fisher trace computation: {e}")

    def on_train_end(self, logs=None):
        """Wait for all pending computations and cleanup"""
        log_and_flush(self.logger, "Waiting for pending Fisher trace computations...")
        for future in self.pending_futures:
            try:
                future.result(timeout=60)  # Wait max 60 seconds per task
            except Exception as e:
                self.logger.error(f"Error waiting for async task: {e}")
        self.executor.shutdown(wait=True)
        log_and_flush(self.logger, "Fisher trace cleanup complete")

    @tf.function  # JIT compile for speed
    def _compute_batch_fisher(self, images, labels):
        """Compute Fisher trace for a single batch"""
        with tf.GradientTape() as tape:
            logits = self.model(images, training=False)
            loss = self.loss_fn(labels, logits)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        # Sum of squared gradients using TensorFlow operations
        batch_fisher = tf.constant(0.0)
        for grad in grads:
            if grad is not None:
                batch_fisher += tf.reduce_sum(tf.square(grad))
        
        return batch_fisher

    def compute_fisher_trace_tf(self):
        """Compute Fisher trace using pure TensorFlow operations"""
        fisher_trace = tf.constant(0.0)
        
        # Create an iterator from the dataset (resets each time)
        batch_count = 0
        for images, labels in self.val_dataset_a:
            if batch_count >= self.num_batches:
                break
            fisher_trace += self._compute_batch_fisher(images, labels)
            batch_count += 1
        
        return fisher_trace


class ActivationSaturationCallback(tf.keras.callbacks.Callback):
    """Monitor ReLU activation saturation - OPTIMIZED WITH ASYNC"""
    
    def __init__(self, val_dataset, epsilon=1e-3, log_dir='logs', num_batches=5, batch_size=32, max_workers=2):
        super(ActivationSaturationCallback, self).__init__()
        # Pre-batch, cache, repeat, and prefetch the dataset for optimal performance
        self.val_dataset = (val_dataset
                           .batch(batch_size)
                           .take(num_batches)
                           .cache()
                           .repeat()
                           .prefetch(tf.data.AUTOTUNE))
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)
        self.log_dir = log_dir
        self.num_batches = num_batches
        self.logger = get_logger('ActivationSaturation', log_dir, 'training.log')
        # Cache intermediate models
        self.intermediate_models = {}
        
        # Threading support
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_futures = []

    def on_epoch_end(self, epoch, logs=None):
        relu_layers = []
        for layer in self.model.layers:
            if hasattr(layer, 'activation') and layer.activation is not None:
                activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
                if 'relu' in activation_name.lower():
                    relu_layers.append(layer)
        
        # Submit async computation for all ReLU layers
        for layer in relu_layers:
            future = self.executor.submit(
                self._compute_saturation_async,
                epoch,
                layer
            )
            self.pending_futures.append(future)
        
        # Clean up completed futures
        self.pending_futures = [f for f in self.pending_futures if not f.done()]

    def _compute_saturation_async(self, epoch, layer):
        """Compute saturation asynchronously in a separate thread"""
        try:
            saturation = self.compute_saturation_tf(layer)
            log_and_flush(self.logger, f"Epoch {epoch + 1}: Layer {layer.name} ReLU Saturation = {saturation.numpy():.6f}")
        except Exception as e:
            self.logger.error(f"Error in async saturation computation for layer {layer.name}: {e}")

    def on_train_end(self, logs=None):
        """Wait for all pending computations and cleanup"""
        log_and_flush(self.logger, "Waiting for pending activation saturation computations...")
        for future in self.pending_futures:
            try:
                future.result(timeout=60)  # Wait max 60 seconds per task
            except Exception as e:
                self.logger.error(f"Error waiting for async task: {e}")
        self.executor.shutdown(wait=True)
        log_and_flush(self.logger, "Activation saturation cleanup complete")

    def get_intermediate_model(self, layer):
        """Cache intermediate models to avoid recreating them"""
        if layer.name not in self.intermediate_models:
            self.intermediate_models[layer.name] = tf.keras.Model(
                inputs=self.model.input, 
                outputs=layer.output
            )
        return self.intermediate_models[layer.name]

    @tf.function  # JIT compile for speed
    def _compute_batch_saturation(self, intermediate_model, images):
        """Compute saturation for a single batch"""
        activations = intermediate_model(images, training=False)
        near_zero = tf.reduce_sum(
            tf.cast(tf.abs(activations) < self.epsilon, tf.float32)
        )
        total = tf.cast(tf.size(activations), tf.float32)
        return near_zero, total

    def compute_saturation_tf(self, layer):
        """Compute saturation using pure TensorFlow operations"""
        intermediate_model = self.get_intermediate_model(layer)
        
        total_near_zero = tf.constant(0.0)
        total_elements = tf.constant(0.0)
        
        # Iterate with manual count to avoid iterator exhaustion
        batch_count = 0
        for images, _ in self.val_dataset:
            if batch_count >= self.num_batches:
                break
            near_zero, total = self._compute_batch_saturation(intermediate_model, images)
            total_near_zero += near_zero
            total_elements += total
            batch_count += 1
        
        return total_near_zero / tf.maximum(total_elements, 1.0)

# ============================================================================
# DOMAIN ADAPTATION CALLBACKS
# ============================================================================

class TargetDomainGapCallback(tf.keras.callbacks.Callback):
    """Track validation performance gap on target domain"""
    
    def __init__(self, val_dataset_b, n_steps=100, k_window=10, ema_alpha=0.9, log_dir='logs'):
        super(TargetDomainGapCallback, self).__init__()
        self.val_dataset_b = val_dataset_b
        self.n_steps = n_steps
        self.k_window = k_window
        self.ema_alpha = ema_alpha
        self.log_dir = log_dir
        
        self.step_counter = 0
        self.best_loss_b = float('inf')
        self.loss_history = []
        self.ema_loss = None
        
        self.logger = get_logger('TargetGap', log_dir, 'domain_adaptation.log')
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    def on_train_begin(self, logs=None):
        self.step_counter = 0
        self.best_loss_b = float('inf')
        self.loss_history = []
        self.ema_loss = None
        #self.logger.info("=" * 60)
        #self.logger.info("Target Domain Gap Tracking Started")
        #self.logger.info("=" * 60)
        log_and_flush(self.logger, "=" * 60)
        log_and_flush(self.logger, "Target Domain Gap Tracking Started")
        log_and_flush(self.logger, "=" * 60)
    
    def on_batch_end(self, batch, logs=None):
        self.step_counter += 1
        
        if self.step_counter % self.n_steps == 0:
            current_loss = self.compute_validation_loss(self.val_dataset_b)
            
            if current_loss < self.best_loss_b:
                self.best_loss_b = current_loss
            
            self.loss_history.append(current_loss)
            if len(self.loss_history) > self.k_window:
                self.loss_history.pop(0)
            
            if self.ema_loss is None:
                self.ema_loss = current_loss
            else:
                self.ema_loss = (self.ema_alpha * self.ema_loss + 
                               (1 - self.ema_alpha) * current_loss)
            
            gap = current_loss - self.best_loss_b
            
            #self.logger.info(
            log_and_flush(self.logger, 
                f"Step {self.step_counter}: "
                f"Loss_B={current_loss:.4f}, "
                f"Best={self.best_loss_b:.4f}, "
                f"EMA={self.ema_loss:.4f}, "
                f"Gap={gap:.4f}"
            )
    
    def compute_validation_loss(self, dataset):
        losses = []
        for images, labels in dataset:
            logits = self.model(images, training=False)
            loss = self.loss_fn(labels, logits)
            losses.append(float(loss))
        return np.mean(losses) if losses else 0.0


class EntropyGapCallback(tf.keras.callbacks.Callback):
    """Measure entropy difference between source and target domains"""
    
    def __init__(self, dataset_a_subset, dataset_b_subset, n_steps=100, batch_size=32, log_dir='logs'):
        super(EntropyGapCallback, self).__init__()
        self.dataset_a_subset = dataset_a_subset
        self.dataset_b_subset = dataset_b_subset
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.log_dir = log_dir
        
        self.step_counter = 0
        self.logger = get_logger('EntropyGap', log_dir, 'domain_adaptation.log')
    
    def on_train_begin(self, logs=None):
        self.step_counter = 0
        #self.logger.info("=" * 60)
        #self.logger.info("Entropy Gap Tracking Started")
        #self.logger.info("=" * 60)
        log_and_flush(self.logger, "=" * 60)
        log_and_flush(self.logger, "Entropy Gap Tracking Started")
        log_and_flush(self.logger, "=" * 60)
    
    def on_batch_end(self, batch, logs=None):
        self.step_counter += 1
        
        if self.step_counter % self.n_steps == 0:
            entropy_a = self.compute_entropy_dataset(self.dataset_a_subset)
            entropy_b = self.compute_entropy_dataset(self.dataset_b_subset)
            gap = entropy_b - entropy_a
            
            #self.logger.info(
            log_and_flush(self.logger, 
                f"Step {self.step_counter}: "
                f"H_A={entropy_a:.4f}, "
                f"H_B={entropy_b:.4f}, "
                f"Gap={gap:.4f}"
            )
    
    def compute_entropy_dataset(self, dataset):
        all_logits = []
        for images, _ in dataset:
            logits = self.model(images, training=False)
            all_logits.append(logits.numpy())
        
        if not all_logits:
            return 0.0
        
        all_logits = np.concatenate(all_logits, axis=0)
        return compute_entropy(all_logits)


class RepresentationMismatchCallback(tf.keras.callbacks.Callback):
    """Measure distribution distance between domains at multiple layers"""
    
    def __init__(self, dataset_a_subset, dataset_b_subset, n_steps=100, 
                 batch_size=32, mmd_kernel='rbf', mmd_gamma=1.0, log_dir='logs'):
        super(RepresentationMismatchCallback, self).__init__()
        self.dataset_a_subset = dataset_a_subset
        self.dataset_b_subset = dataset_b_subset
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.mmd_kernel = mmd_kernel
        self.mmd_gamma = mmd_gamma
        self.log_dir = log_dir
        
        self.step_counter = 0
        self.intermediate_models = {}
        self.layer_names = {}
        
        self.logger = get_logger('RepMismatch', log_dir, 'domain_adaptation.log')
    
    def on_train_begin(self, logs=None):
        self.step_counter = 0
        self.identify_layer_positions()
        self.create_intermediate_models()
        
        #self.logger.info("=" * 60)
        #self.logger.info("Representation Mismatch Tracking Started")
        #self.logger.info(f"Monitoring layers: {list(self.layer_names.values())}")
        #self.logger.info("=" * 60)
        log_and_flush(self.logger, "=" * 60)
        log_and_flush(self.logger, "Representation Mismatch Tracking Started")
        log_and_flush(self.logger, f"Monitoring layers: {list(self.layer_names.values())}")
        log_and_flush(self.logger, "=" * 60)
    
    def on_batch_end(self, batch, logs=None):
        self.step_counter += 1
        
        if self.step_counter % self.n_steps == 0:
            #self.logger.info(f"Step {self.step_counter} - Computing representation distances...")
            log_and_flush(self.logger, f"Step {self.step_counter} - Computing representation distances...")
            
            for position, layer_name in self.layer_names.items():
                embeddings_a = self.extract_embeddings(self.dataset_a_subset, position)
                embeddings_b = self.extract_embeddings(self.dataset_b_subset, position)
                
                mmd = compute_mmd(embeddings_a, embeddings_b, 
                                 kernel=self.mmd_kernel, gamma=self.mmd_gamma)
                kl = compute_kl_divergence(embeddings_a, embeddings_b)
                wasserstein = compute_wasserstein_distance(embeddings_a, embeddings_b)
                
                #self.logger.info(
                log_and_flush(self.logger, 
                    f"  Layer={layer_name} ({position}): "
                    f"MMD={mmd:.4f}, KL={kl:.4f}, Wasserstein={wasserstein:.4f}"
                )
    
    def identify_layer_positions(self):
        weighted_layers = [
            layer for layer in self.model.layers
            if len(layer.weights) > 0 and not isinstance(
                layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)
            )
        ]
        
        if len(weighted_layers) < 3:
            raise ValueError("Model must have at least 3 weighted layers")
        
        self.layer_names['first'] = weighted_layers[0].name
        self.layer_names['middle'] = weighted_layers[len(weighted_layers) // 2].name
        self.layer_names['penultimate'] = weighted_layers[-2].name
    
    def create_intermediate_models(self):
        for position, layer_name in self.layer_names.items():
            layer = self.model.get_layer(layer_name)
            self.intermediate_models[position] = tf.keras.Model(
                inputs=self.model.input,
                outputs=layer.output
            )
    
    def extract_embeddings(self, dataset, position):
        embeddings = []
        for images, _ in dataset:
            layer_output = self.intermediate_models[position](images, training=False)
            if len(layer_output.shape) > 2:
                batch_size = tf.shape(layer_output)[0]
                layer_output = tf.reshape(layer_output, [batch_size, -1])
            embeddings.append(layer_output.numpy())
        
        if not embeddings:
            return np.array([])
        return np.concatenate(embeddings, axis=0)


# ============================================================================
# UNIFIED CALLBACK FACTORY
# ============================================================================

class CallbackFactory:
    """
    Unified factory for creating all callbacks from configuration
    Supports both convergence monitoring and domain adaptation callbacks
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize factory with configuration file
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self.load_config(config_path)
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    # ------------------------------------------------------------------------
    # Standard Keras Callbacks
    # ------------------------------------------------------------------------
    
    def create_checkpoint(self, filepath='best_model.h5', monitor='val_loss', 
                         mode='min', save_best_only=True, verbose=1):
        """Create ModelCheckpoint callback"""
        return ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
            verbose=verbose
        )
    
    def create_early_stopping(self, monitor='val_loss', mode='min', 
                             patience=5, restore_best_weights=True, verbose=1):
        """Create EarlyStopping callback"""
        return EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=verbose
        )
    
    def create_tensorboard(self, log_dir='logs/tensorboard', histogram_freq=1, 
                          write_graph=True, update_freq='epoch'):
        """Create TensorBoard callback"""
        return tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            write_graph=write_graph,
            update_freq=update_freq
        )
    
    # ------------------------------------------------------------------------
    # Convergence Monitoring Callbacks
    # ------------------------------------------------------------------------
    
    def create_training_metrics_logger(self, log_dir='logs'):
        """Create TrainingMetricsLogger callback"""
        return TrainingMetricsLogger(log_dir=log_dir)
    
    def create_gradient_analysis(self, log_dir='logs', K=50, ema_alpha=0.99, epsilon=1e-8, max_workers=2):
        """Create GradientAnalysisCallback"""
        return GradientAnalysisCallback(
            log_dir=log_dir,
            K=K,
            ema_alpha=ema_alpha,
            epsilon=epsilon,
            max_workers=max_workers
        )
    
    def create_fisher_trace(self, val_dataset_a, n_steps=100, log_dir='logs', max_workers=1):
        """Create FisherTraceCallback"""
        return FisherTraceCallback(
            val_dataset_a=val_dataset_a,
            n_steps=n_steps,
            log_dir=log_dir,
            max_workers=max_workers
        )
    
    def create_activation_saturation(self, val_dataset, epsilon=1e-3, log_dir='logs', max_workers=2):
        """Create ActivationSaturationCallback"""
        return ActivationSaturationCallback(
            val_dataset=val_dataset,
            epsilon=epsilon,
            log_dir=log_dir,
            max_workers=max_workers
        )
    
    # ------------------------------------------------------------------------
    # Domain Adaptation Callbacks
    # ------------------------------------------------------------------------
    
    def create_target_domain_gap(self, val_dataset_b, n_steps=100, k_window=10, 
                                ema_alpha=0.9, log_dir='logs'):
        """Create TargetDomainGapCallback"""
        return TargetDomainGapCallback(
            val_dataset_b=val_dataset_b,
            n_steps=n_steps,
            k_window=k_window,
            ema_alpha=ema_alpha,
            log_dir=log_dir
        )
    
    def create_entropy_gap(self, dataset_a_subset, dataset_b_subset, n_steps=100, 
                          batch_size=32, log_dir='logs'):
        """Create EntropyGapCallback"""
        return EntropyGapCallback(
            dataset_a_subset=dataset_a_subset,
            dataset_b_subset=dataset_b_subset,
            n_steps=n_steps,
            batch_size=batch_size,
            log_dir=log_dir
        )
    
    def create_representation_mismatch(self, dataset_a_subset, dataset_b_subset, 
                                      n_steps=100, batch_size=32, mmd_kernel='rbf', 
                                      mmd_gamma=1.0, log_dir='logs'):
        """Create RepresentationMismatchCallback"""
        return RepresentationMismatchCallback(
            dataset_a_subset=dataset_a_subset,
            dataset_b_subset=dataset_b_subset,
            n_steps=n_steps,
            batch_size=batch_size,
            mmd_kernel=mmd_kernel,
            mmd_gamma=mmd_gamma,
            log_dir=log_dir
        )
    
    # ------------------------------------------------------------------------
    # Batch Creation Methods
    # ------------------------------------------------------------------------
    
    def create_all_convergence_callbacks(self, val_dataset, log_dir='logs'):
        """
        Create all convergence monitoring callbacks
        
        Args:
            val_dataset: Validation dataset
            test_dataset: Test dataset
            log_dir: Directory for logs
        
        Returns:
            List of convergence monitoring callbacks
        """
        return [
            self.create_gradient_analysis(log_dir=log_dir, K=50, ema_alpha=0.99),
            self.create_fisher_trace(val_dataset, n_steps=100, log_dir=log_dir),
            self.create_activation_saturation(val_dataset, epsilon=1e-3, log_dir=log_dir)
        ]
    
    def create_all_domain_adaptation_callbacks(self, dataset_a_subset, dataset_b_subset, 
                                               n_steps=100, log_dir='logs'):
        """
        Create all domain adaptation callbacks
        
        Args:
            dataset_a_subset: Held-out subset from source domain
            dataset_b_subset: Held-out subset from target domain
            n_steps: Compute metrics every N steps
            log_dir: Directory for logs
        
        Returns:
            List of domain adaptation callbacks
        """
        da_config = self.config.get('domain_adaptation', {})
        
        return [
            self.create_target_domain_gap(
                val_dataset_b=dataset_b_subset,
                n_steps=n_steps,
                k_window=da_config.get('target_gap', {}).get('k_window', 10),
                ema_alpha=da_config.get('target_gap', {}).get('ema_alpha', 0.9),
                log_dir=log_dir
            ),
            self.create_entropy_gap(
                dataset_a_subset=dataset_a_subset,
                dataset_b_subset=dataset_b_subset,
                n_steps=n_steps,
                batch_size=da_config.get('entropy_gap', {}).get('batch_size', 32),
                log_dir=log_dir
            ),
            self.create_representation_mismatch(
                dataset_a_subset=dataset_a_subset,
                dataset_b_subset=dataset_b_subset,
                n_steps=n_steps,
                batch_size=da_config.get('representation_mismatch', {}).get('batch_size', 32),
                mmd_kernel=da_config.get('representation_mismatch', {}).get('mmd_kernel', 'rbf'),
                mmd_gamma=da_config.get('representation_mismatch', {}).get('mmd_gamma', 1.0),
                log_dir=log_dir
            )
        ]
    
    def create_standard_callbacks(self, checkpoint_path='checkpoints/best_model.h5',
                                  monitor='val_accuracy', patience=10):
        """
        Create standard training callbacks (checkpoint, early stopping, tensorboard, metrics logging)
        
        Args:
            checkpoint_path: Path to save best model
            monitor: Metric to monitor for checkpoint/early stopping
            patience: Patience for early stopping
        
        Returns:
            List of standard callbacks
        """
        return [
            self.create_training_metrics_logger(log_dir='logs'),
            self.create_checkpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                mode='min',
                save_best_only=True
            ),
            self.create_early_stopping(
                monitor='val_loss',
                mode='min',
                patience=patience,
                restore_best_weights=True
            ),
            self.create_tensorboard(
                log_dir='logs/tensorboard',
                histogram_freq=1
            )
        ]
