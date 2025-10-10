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
    return logger


# ============================================================================
# CONVERGENCE MONITORING CALLBACKS
# ============================================================================

class GradientAnalysisCallback(tf.keras.callbacks.Callback):
    """Analyze gradient statistics: norm, SNR, cosine similarity"""
    
    def __init__(self, log_dir='logs', K=50, ema_alpha=0.99, epsilon=1e-8):
        super(GradientAnalysisCallback, self).__init__()
        self.log_dir = log_dir
        self.K = K
        self.ema_alpha = ema_alpha
        self.epsilon = epsilon
        self.gradients_history = []
        self.ema_cosine_similarity = None
        self.logger = get_logger('GradientAnalysis', log_dir, 'training.log')

    def on_train_begin(self, logs=None):
        self.gradients_history = []
        self.ema_cosine_similarity = None

    def on_batch_end(self, batch, logs=None):
        if not hasattr(self.model, 'current_gradients') or self.model.current_gradients is None:
            return
        
        gradients = self.flatten_gradients(self.model.current_gradients)
        global_gradient_norm = tf.norm(gradients, ord=2).numpy()
        self.logger.info(f"Batch {batch + 1}: Global Gradient Norm = {global_gradient_norm}")

        self.gradients_history.append(gradients.numpy())
        if len(self.gradients_history) > self.K:
            self.gradients_history.pop(0)

        if len(self.gradients_history) == self.K:
            snr = self.compute_snr(self.gradients_history)
            self.logger.info(f"Batch {batch + 1}: SNR = {snr}")

        if len(self.gradients_history) > 1:
            cosine_similarity = self.compute_cosine_similarity(
                self.gradients_history[-1], 
                self.gradients_history[-2]
            )
            if self.ema_cosine_similarity is None:
                self.ema_cosine_similarity = cosine_similarity
            else:
                self.ema_cosine_similarity = (self.ema_alpha * self.ema_cosine_similarity + 
                                             (1 - self.ema_alpha) * cosine_similarity)
            self.logger.info(f"Batch {batch + 1}: EMA Cosine Similarity = {self.ema_cosine_similarity}")

    def flatten_gradients(self, gradients):
        flattened = []
        for grad in gradients:
            if grad is not None:
                flattened.append(tf.reshape(grad, [-1]))
        return tf.concat(flattened, axis=0)

    def compute_snr(self, gradients_history):
        gradients_array = np.array(gradients_history)
        mean_gradient = np.mean(gradients_array, axis=0)
        variance_gradient = np.mean((gradients_array - mean_gradient) ** 2, axis=0)
        snr = np.linalg.norm(mean_gradient) / np.sqrt(np.sum(variance_gradient) + self.epsilon)
        return snr

    def compute_cosine_similarity(self, grad1, grad2):
        dot_product = np.sum(grad1 * grad2)
        norm_grad1 = np.linalg.norm(grad1)
        norm_grad2 = np.linalg.norm(grad2)
        cosine_similarity = dot_product / (norm_grad1 * norm_grad2 + self.epsilon)
        return cosine_similarity


class FisherTraceCallback(tf.keras.callbacks.Callback):
    """Compute Fisher Information trace on validation sets"""
    
    def __init__(self, val_dataset_a, val_dataset_b, n_steps=100, log_dir='logs'):
        super(FisherTraceCallback, self).__init__()
        self.val_dataset_a = val_dataset_a
        self.val_dataset_b = val_dataset_b
        self.n_steps = n_steps
        self.log_dir = log_dir
        self.step_counter = 0
        self.logger = get_logger('FisherTrace', log_dir, 'training.log')

    def on_train_begin(self, logs=None):
        self.step_counter = 0

    def on_batch_end(self, batch, logs=None):
        self.step_counter += 1
        if self.step_counter % self.n_steps == 0:
            fisher_trace_a = self.compute_fisher_trace(self.val_dataset_a)
            fisher_trace_b = self.compute_fisher_trace(self.val_dataset_b)
            self.logger.info(f"Step {self.step_counter}: Fisher Trace (Val Set A) = {fisher_trace_a}")
            self.logger.info(f"Step {self.step_counter}: Fisher Trace (Val Set B) = {fisher_trace_b}")

    def compute_fisher_trace(self, dataset):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        fisher_trace = 0.0
        
        for images, labels in dataset.batch(32).take(10):
            with tf.GradientTape() as tape:
                logits = self.model(images, training=False)
                loss = loss_fn(labels, logits)
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            for grad in grads:
                if grad is not None:
                    fisher_trace += tf.reduce_sum(tf.square(grad)).numpy()
        
        return fisher_trace


class ActivationSaturationCallback(tf.keras.callbacks.Callback):
    """Monitor ReLU activation saturation"""
    
    def __init__(self, val_dataset, epsilon=1e-3, log_dir='logs'):
        super(ActivationSaturationCallback, self).__init__()
        self.val_dataset = val_dataset
        self.epsilon = epsilon
        self.log_dir = log_dir
        self.logger = get_logger('ActivationSaturation', log_dir, 'training.log')

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, 'activation') and layer.activation is not None:
                activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
                
                if 'relu' in activation_name.lower():
                    saturation = self.compute_saturation(layer)
                    self.logger.info(f"Epoch {epoch + 1}: Layer {layer.name} ReLU Saturation = {saturation}")

    def compute_saturation(self, layer):
        intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
        
        total_near_zero = 0
        total_elements = 0
        
        for images, _ in self.val_dataset.batch(32).take(5):
            activations = intermediate_model(images, training=False)
            total_near_zero += tf.reduce_sum(tf.cast(tf.abs(activations) < self.epsilon, tf.float32)).numpy()
            total_elements += tf.size(activations).numpy()
        
        return total_near_zero / total_elements if total_elements > 0 else 0.0


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
        self.logger.info("=" * 60)
        self.logger.info("Target Domain Gap Tracking Started")
        self.logger.info("=" * 60)
    
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
            
            self.logger.info(
                f"Step {self.step_counter}: "
                f"Loss_B={current_loss:.4f}, "
                f"Best={self.best_loss_b:.4f}, "
                f"EMA={self.ema_loss:.4f}, "
                f"Gap={gap:.4f}"
            )
    
    def compute_validation_loss(self, dataset):
        losses = []
        for images, labels in dataset.batch(32):
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
        self.logger.info("=" * 60)
        self.logger.info("Entropy Gap Tracking Started")
        self.logger.info("=" * 60)
    
    def on_batch_end(self, batch, logs=None):
        self.step_counter += 1
        
        if self.step_counter % self.n_steps == 0:
            entropy_a = self.compute_entropy_dataset(self.dataset_a_subset)
            entropy_b = self.compute_entropy_dataset(self.dataset_b_subset)
            gap = entropy_b - entropy_a
            
            self.logger.info(
                f"Step {self.step_counter}: "
                f"H_A={entropy_a:.4f}, "
                f"H_B={entropy_b:.4f}, "
                f"Gap={gap:.4f}"
            )
    
    def compute_entropy_dataset(self, dataset):
        all_logits = []
        for images, _ in dataset.batch(self.batch_size):
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
        
        self.logger.info("=" * 60)
        self.logger.info("Representation Mismatch Tracking Started")
        self.logger.info(f"Monitoring layers: {list(self.layer_names.values())}")
        self.logger.info("=" * 60)
    
    def on_batch_end(self, batch, logs=None):
        self.step_counter += 1
        
        if self.step_counter % self.n_steps == 0:
            self.logger.info(f"Step {self.step_counter} - Computing representation distances...")
            
            for position, layer_name in self.layer_names.items():
                embeddings_a = self.extract_embeddings(self.dataset_a_subset, position)
                embeddings_b = self.extract_embeddings(self.dataset_b_subset, position)
                
                mmd = compute_mmd(embeddings_a, embeddings_b, 
                                 kernel=self.mmd_kernel, gamma=self.mmd_gamma)
                kl = compute_kl_divergence(embeddings_a, embeddings_b)
                wasserstein = compute_wasserstein_distance(embeddings_a, embeddings_b)
                
                self.logger.info(
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
        for images, _ in dataset.batch(self.batch_size):
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
    
    def create_gradient_analysis(self, log_dir='logs', K=50, ema_alpha=0.99, epsilon=1e-8):
        """Create GradientAnalysisCallback"""
        return GradientAnalysisCallback(
            log_dir=log_dir,
            K=K,
            ema_alpha=ema_alpha,
            epsilon=epsilon
        )
    
    def create_fisher_trace(self, val_dataset_a, val_dataset_b, n_steps=100, log_dir='logs'):
        """Create FisherTraceCallback"""
        return FisherTraceCallback(
            val_dataset_a=val_dataset_a,
            val_dataset_b=val_dataset_b,
            n_steps=n_steps,
            log_dir=log_dir
        )
    
    def create_activation_saturation(self, val_dataset, epsilon=1e-3, log_dir='logs'):
        """Create ActivationSaturationCallback"""
        return ActivationSaturationCallback(
            val_dataset=val_dataset,
            epsilon=epsilon,
            log_dir=log_dir
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
    
    def create_all_convergence_callbacks(self, val_dataset, test_dataset, log_dir='logs'):
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
            self.create_fisher_trace(val_dataset, test_dataset, n_steps=100, log_dir=log_dir),
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
        Create standard training callbacks (checkpoint, early stopping, tensorboard)
        
        Args:
            checkpoint_path: Path to save best model
            monitor: Metric to monitor for checkpoint/early stopping
            patience: Patience for early stopping
        
        Returns:
            List of standard callbacks
        """
        return [
            self.create_checkpoint(
                filepath=checkpoint_path,
                monitor=monitor,
                mode='max',
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
