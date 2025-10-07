#!/usr/bin/env python3
"""
Main script to orchestrate CNN training on REAL dataset
"""

import os
import yaml
import tensorflow as tf
from model_creator import ModelCreator
from load_and_create_datasets import DataLoader
from trainer import Trainer
from callback_creator import (
    CallbackCreator, 
    GradientAnalysisCallback, 
    FisherTraceCallback, 
    ActivationSaturationCallback
)

def load_training_config(config_path='config.yaml'):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_directories():
    """Create necessary directories for saving models and logs"""
    directories = ['models', 'logs', 'checkpoints']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    print("=" * 60)
    print("CNN Training Pipeline - REAL Dataset")
    print("=" * 60)
    
    # Configuration
    CONFIG_PATH = 'config.yaml'
    DATA_ROOT_DIR = './data'  # Update this to your actual data directory
    MODEL_TYPE = 'CNN'
    
    # Setup directories
    print("\n[1/7] Setting up directories...")
    setup_directories()
    
    # Load configuration
    print("\n[2/7] Loading configuration...")
    config = load_training_config(CONFIG_PATH)
    training_config = config.get('training', {})
    
    EPOCHS = training_config.get('epochs', 50)
    BATCH_SIZE = training_config.get('batch_size', 32)
    OPTIMIZER = training_config.get('optimizer', 'adam')
    LEARNING_RATE = training_config.get('learning_rate', 0.001)
    TRAIN_SPLIT = training_config.get('train_split', 0.7)
    VAL_SPLIT = training_config.get('val_split', 0.15)
    TEST_SPLIT = training_config.get('test_split', 0.15)
    
    print(f"  - Model Type: {MODEL_TYPE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Optimizer: {OPTIMIZER}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    
    # Load and prepare datasets
    print("\n[3/7] Loading REAL dataset...")
    data_loader = DataLoader(root_dir=DATA_ROOT_DIR)
    
    try:
        # Create dataset for REAL images
        real_dataset = data_loader.create_dataset(image_type='REAL')
        
        # Split dataset
        print("  - Splitting dataset...")
        train_dataset, val_dataset, test_dataset = data_loader.split_dataset(
            real_dataset,
            train_split=TRAIN_SPLIT,
            val_split=VAL_SPLIT,
            test_split=TEST_SPLIT
        )
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Cache and prefetch for performance
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    print("  ✓ Dataset loaded and split successfully")
    
    # Create model
    print(f"\n[4/7] Creating {MODEL_TYPE} model...")
    model_creator = ModelCreator(config_path=CONFIG_PATH)
    model = model_creator.create_model(MODEL_TYPE)
    
    print(f"  ✓ Model created successfully")
    print(f"  - Total parameters: {model.count_params():,}")
    
    # Display model summary
    print("\nModel Summary:")
    model.summary()
    
    # Create callbacks
    print("\n[5/7] Setting up callbacks...")
    callback_creator = CallbackCreator(config_path=CONFIG_PATH)
    
    callbacks = [
        # Model checkpoint - save best model
        callback_creator.create_checkpoint_callback(
            filepath='checkpoints/best_cnn_real.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        ),
        
        # Early stopping
        callback_creator.create_early_stopping_callback(
            monitor='val_loss',
            mode='min',
            patience=10,
            restore_best_weights=True
        ),
        
        # Gradient analysis callback
        GradientAnalysisCallback(
            log_dir='logs',
            K=50,
            ema_alpha=0.99
        ),
        
        # Fisher trace callback
        FisherTraceCallback(
            val_dataset_a=val_dataset,
            val_dataset_b=test_dataset,
            n_steps=100,
            log_dir='logs'
        ),
        
        # Activation saturation callback
        ActivationSaturationCallback(
            val_dataset=val_dataset,
            epsilon=1e-3,
            log_dir='logs'
        ),
        
        # TensorBoard callback
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    print(f"  ✓ {len(callbacks)} callbacks configured")
    
    # Create trainer (datasets already batched)
    print("\n[6/7] Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset.unbatch(),  # Unbatch for trainer
        val_dataset=val_dataset.unbatch(),
        test_dataset=test_dataset.unbatch(),
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        callbacks=callbacks
    )
    print("  ✓ Trainer initialized")
    
    # Train model
    print("\n[7/7] Starting training...")
    print("=" * 60)
    
    try:
        history = trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = trainer.evaluate(batch_size=BATCH_SIZE)
        print(f"\nTest Results:")
        print(f"  - Test Loss: {test_loss:.4f}")
        print(f"  - Test Accuracy: {test_accuracy:.4f}")
        
        # Save final model
        final_model_path = 'models/final_cnn_real.h5'
        model.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")
        
        # Print training summary
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print("\nLogs saved to: ./logs/")
        print("TensorBoard logs: ./logs/tensorboard/")
        print("Best model checkpoint: ./checkpoints/best_cnn_real.h5")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model state...")
        model.save('models/interrupted_cnn_real.h5')
        print("✓ Model saved to: models/interrupted_cnn_real.h5")
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
