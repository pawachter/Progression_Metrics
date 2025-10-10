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
    