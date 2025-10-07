import tensorflow as tf
from tensorflow.keras.callbacks import CallbackList
import numpy as np

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, optimizer='sgd', learning_rate=0.01, callbacks=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        if optimizer.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            self.optimizer = tf.keras.optimizers.get(optimizer)
        
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        
        self.callbacks = CallbackList(callbacks if callbacks is not None else [], model=model)
        self.model.current_gradients = None  # Initialize on model, not self

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.loss_fn(y, logits)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.current_gradients = gradients  # Store on model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_acc_metric.update_state(y, logits)
        return loss

    def val_step(self, x, y):
        logits = self.model(x, training=False)
        loss = self.loss_fn(y, logits)
        self.val_acc_metric.update_state(y, logits)
        return loss

    def train(self, epochs=10, batch_size=32):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])
        self.model.stop_training = False  # Initialize stop_training flag
        
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        train_ds = self.train_dataset.batch(batch_size)
        val_ds = self.val_dataset.batch(batch_size)
        
        self.callbacks.on_train_begin()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            self.callbacks.on_epoch_begin(epoch)
            
            self.train_acc_metric.reset_state()
            epoch_loss = []
            
            for batch_idx, (x_batch, y_batch) in enumerate(train_ds):
                self.callbacks.on_batch_begin(batch_idx)
                
                loss = self.train_step(x_batch, y_batch)
                epoch_loss.append(loss)
                
                logs = {'loss': float(loss), 'accuracy': float(self.train_acc_metric.result())}
                self.callbacks.on_batch_end(batch_idx, logs)
                
                if batch_idx % 50 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss:.4f}, Acc: {self.train_acc_metric.result():.4f}")
            
            train_loss = np.mean(epoch_loss)
            train_acc = float(self.train_acc_metric.result())
            
            self.val_acc_metric.reset_state()
            val_losses = []
            for x_batch, y_batch in val_ds:
                val_loss = self.val_step(x_batch, y_batch)
                val_losses.append(val_loss)
            
            val_loss = np.mean(val_losses)
            val_acc = float(self.val_acc_metric.result())
            
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            print(f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            logs = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            self.callbacks.on_epoch_end(epoch, logs)
            
            if self.model.stop_training:
                print("Early stopping triggered")
                break
        
        self.callbacks.on_train_end()
        
        class History:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return History(history)

    def evaluate(self, batch_size=32):
        self.val_acc_metric.reset_state()
        test_ds = self.test_dataset.batch(batch_size)
        test_losses = []
        
        for x_batch, y_batch in test_ds:
            loss = self.val_step(x_batch, y_batch)
            test_losses.append(loss)
        
        return np.mean(test_losses), float(self.val_acc_metric.result())
