import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import logging
import yaml
import os

# Create a shared logger to avoid conflicts
def get_logger(name, log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:  # Only add handler if not already configured
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    return logger


class GradientAnalysisCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir='logs', K=50, ema_alpha=0.99, epsilon=1e-8):
        super(GradientAnalysisCallback, self).__init__()
        self.log_dir = log_dir
        self.K = K
        self.ema_alpha = ema_alpha
        self.epsilon = epsilon
        self.gradients_history = []
        self.ema_cosine_similarity = None
        self.logger = get_logger('GradientAnalysis', log_dir)

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
    def __init__(self, val_dataset_a, val_dataset_b, n_steps=100, log_dir='logs'):
        super(FisherTraceCallback, self).__init__()
        self.val_dataset_a = val_dataset_a
        self.val_dataset_b = val_dataset_b
        self.n_steps = n_steps
        self.log_dir = log_dir
        self.step_counter = 0
        self.logger = get_logger('FisherTrace', log_dir)

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
    def __init__(self, val_dataset, epsilon=1e-3, log_dir='logs'):
        super(ActivationSaturationCallback, self).__init__()
        self.val_dataset = val_dataset
        self.epsilon = epsilon
        self.log_dir = log_dir
        self.logger = get_logger('ActivationSaturation', log_dir)

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


class CallbackCreator:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('callbacks', {})

    def create_checkpoint_callback(self, filepath='best_model.h5', monitor='val_loss', mode='min', save_best_only=True):
        return ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
            verbose=1
        )

    def create_early_stopping_callback(self, monitor='val_loss', mode='min', patience=5, restore_best_weights=True):
        return EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=1
        )
