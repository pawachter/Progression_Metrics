#!/usr/bin/env python3
"""
Main script to orchestrate CNN training on REAL dataset with Domain Adaptation
Uses unified CallbackFactory for all callbacks
"""

import os
import yaml
import tensorflow as tf
from model_creator import ModelCreator
from load_and_create_datasets import DataLoader
from trainer import Trainer
from callbacks import CallbackFactory

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
    print("CNN Training Pipeline - REAL Dataset with Domain Adaptation")
    print("=" * 60)
    
    # Configuration
    CONFIG_PATH = 'config.yaml'
    DATA_ROOT_DIR = './data'  # Update this to your actual data directory
    MODEL_TYPE = 'CNN'
    
    # Setup directories
    print("\n[1/8] Setting up directories...")
    setup_directories()
    
    # Load configuration
    print("\n[2/8] Loading configuration...")
    config = load_training_config(CONFIG_PATH)
    training_config = config.get('training', {})
    da_config = config.get('domain_adaptation', {})
    
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
    print(f"  - Domain Adaptation: {'Enabled' if da_config.get('enabled', False) else 'Disabled'}")
    
    # Load and prepare datasets
    print("\n[3/8] Loading datasets...")
    data_loader = DataLoader(root_dir=DATA_ROOT_DIR)
    
    try:
        # Create datasets for both domains
        dataset_a = data_loader.create_dataset(image_type='FAKE')  # Source domain
        dataset_b = data_loader.create_dataset(image_type='REAL')  # Target domain
        
        # Create held-out subsets for domain adaptation (if enabled)
        dataset_a_subset = None
        dataset_b_subset = None
        
        if da_config.get('enabled', False):
            print("\n[3b/8] Creating held-out subsets for domain adaptation...")
            subset_size = da_config.get('subset_size', 500)
            
            dataset_a_subset = dataset_a.take(subset_size).cache()
            dataset_b_subset = dataset_b.take(subset_size).cache()
            
            # Skip held-out data for training
            dataset_a = dataset_a.skip(subset_size)
            dataset_b = dataset_b.skip(subset_size)
            
            print(f"  ✓ Created held-out subsets: {subset_size} samples each")
        
        # Use target domain (REAL) for training
        real_dataset = dataset_b
        
        # Split dataset
        print("  - Splitting dataset...")
        train_dataset, val_dataset, test_dataset = data_loader.split_dataset(
            real_dataset,
            train_split=TRAIN_SPLIT,
            val_split=VAL_SPLIT,
            test_split=TEST_SPLIT
        )
        
        print(f"  ✓ Datasets created successfully")
        
    except ValueError as e:
        print(f"  ✗ Error loading datasets: {e}")
        print(f"  Please ensure data is in {DATA_ROOT_DIR}")
        return
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return
    
    # Create model
    print("\n[4/8] Creating model...")
    model_creator = ModelCreator(CONFIG_PATH)
    model = model_creator.create_model(MODEL_TYPE)
    model.summary()
    print(f"  ✓ Model created: {MODEL_TYPE}")
    
    # Create callbacks
    print("\n[5/8] Initializing callbacks...")
    callback_factory = CallbackFactory(CONFIG_PATH)
    
    # Standard callbacks
    standard_callbacks = callback_factory.create_standard_callbacks(
        checkpoint_path='checkpoints/best_model.h5',
        monitor='val_loss',
        patience=training_config.get('early_stopping_patience', 10)
    )
    print(f"  ✓ Standard callbacks: Checkpoint, EarlyStopping, TensorBoard")
    
    # Convergence monitoring callbacks
    convergence_callbacks = callback_factory.create_all_convergence_callbacks(
        val_dataset=val_dataset,
        #test_dataset=test_dataset,
        log_dir='logs'
    )
    print(f"  ✓ Convergence callbacks: GradientAnalysis, FisherTrace, ActivationSaturation")
    
    # Domain adaptation callbacks (if enabled)
    all_callbacks = standard_callbacks + convergence_callbacks
    
    if da_config.get('enabled', False) and dataset_a_subset is not None and dataset_b_subset is not None:
        da_callbacks = callback_factory.create_all_domain_adaptation_callbacks(
            dataset_a_subset=dataset_a_subset,
            dataset_b_subset=dataset_b_subset,
            n_steps=da_config.get('n_steps', 100),
            log_dir='logs'
        )
        all_callbacks += da_callbacks
        print(f"  ✓ Domain Adaptation callbacks: TargetGap, EntropyGap, RepresentationMismatch")
    else:
        print(f"  ⊘ Domain Adaptation callbacks: Disabled")
    
    print(f"  Total callbacks: {len(all_callbacks)}")
    
    # Create trainer
    print("\n[6/8] Creating trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        callbacks=all_callbacks
    )
    print(f"  ✓ Trainer initialized")
    
    # Train model
    print("\n[7/8] Starting training...")
    print("=" * 60)
    try:
        history = trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE)
        print("=" * 60)
        print(f"  ✓ Training completed successfully")
    except KeyboardInterrupt:
        print("\n  ⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate model
    print("\n[8/8] Evaluating model on test set...")
    try:
        test_loss, test_accuracy = trainer.evaluate(batch_size=BATCH_SIZE)
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  ✓ Evaluation completed")
    except Exception as e:
        print(f"  ✗ Evaluation failed: {e}")
    
    # Save final model
    print("\nSaving final model...")
    try:
        model.save('models/final_model.h5')
        print(f"  ✓ Model saved to models/final_model.h5")
    except Exception as e:
        print(f"  ✗ Failed to save model: {e}")
    
    print("\n" + "=" * 60)
    print("Training Pipeline Completed")
    print("=" * 60)
    print(f"Logs saved to: logs/")
    print(f"Best model saved to: checkpoints/best_model.h5")
    print(f"Final model saved to: models/final_model.h5")


if __name__ == '__main__':
    main()
    