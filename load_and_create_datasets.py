import tensorflow as tf
import os

class DataLoader:
    """
    DataLoader using TensorFlow's recommended image_dataset_from_directory API.
    This is simpler, faster, and follows TensorFlow best practices.
    """
    
    def __init__(self, root_dir, image_size=(32, 32), batch_size=32):
        """
        Initialize DataLoader with dataset configuration.
        
        Args:
            root_dir: Root directory containing REAL and FAKE subdirectories
            image_size: Target image size (height, width)
            batch_size: Default batch size for loading
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size

    def create_dataset(self, image_type, shuffle=True, seed=42):
        """
        Create dataset using TensorFlow's image_dataset_from_directory.
        
        This function automatically:
        - Infers class labels from subdirectory names
        - Handles image loading and decoding
        - Provides efficient batching and prefetching
        - Supports data augmentation pipelines
        
        Directory structure:
        root_dir/
        ├── REAL/
        │   ├── class_0/*.jpg
        │   ├── class_1/*.jpg
        │   └── ...
        └── FAKE/
            ├── class_0/*.jpg
            ├── class_1/*.jpg
            └── ...
        
        Args:
            image_type: Either 'FAKE' or 'REAL'
            shuffle: Whether to shuffle the dataset
            seed: Random seed for reproducibility
        
        Returns:
            tf.data.Dataset of (image, label) tuples (unbatched)
        """
        if image_type not in ['FAKE', 'REAL']:
            raise ValueError("image_type must be either 'FAKE' or 'REAL'")
        
        directory = os.path.join(self.root_dir, image_type)
        
        # Check if directory exists
        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        # Use TensorFlow's built-in function - follows all best practices
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels='inferred',  # Infer labels from subdirectory structure
            label_mode='int',   # Return integer labels (0-9)
            class_names=None,   # Auto-detect class names from folder names
            color_mode='rgb',
            batch_size=None,    # Return unbatched dataset for flexibility
            image_size=self.image_size,
            shuffle=shuffle,
            seed=seed,
            interpolation='bilinear',
            follow_links=False,
        )
        
        # Normalize images to [0, 1] range
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return dataset

    def split_dataset(self, dataset, train_split=0.7, val_split=0.15, test_split=0.15, seed=42):
        """
        Split dataset into train, validation, and test sets.
        
        Uses TensorFlow's recommended approach with cardinality checking
        and deterministic splitting for reproducibility.
        
        Args:
            dataset: Input tf.data.Dataset
            train_split: Proportion for training (0-1)
            val_split: Proportion for validation (0-1)
            test_split: Proportion for testing (0-1)
            seed: Random seed for shuffling
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Validate splits
        total = train_split + val_split + test_split
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Splits must sum to 1.0, got {total}")
        
        # Get dataset size
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        
        if dataset_size == 0 or dataset_size == tf.data.UNKNOWN_CARDINALITY:
            # Fallback: count elements if cardinality is unknown
            dataset_size = dataset.reduce(0, lambda x, _: x + 1).numpy()
        
        if dataset_size == 0:
            raise ValueError("Dataset is empty")
        
        # Shuffle with seed for reproducibility
        dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed, reshuffle_each_iteration=False)
        
        # Calculate split sizes
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        
        # Split dataset
        train_dataset = dataset.take(train_size)
        remaining = dataset.skip(train_size)
        val_dataset = remaining.take(val_size)
        test_dataset = remaining.skip(val_size)
        
        return train_dataset, val_dataset, test_dataset