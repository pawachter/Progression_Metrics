import tensorflow as tf
import os

class DataLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def get_label_from_filename(self, filename):
        filename = tf.strings.split(filename, os.path.sep)[-1]
        if tf.strings.regex_full_match(filename, r'.*\(\d+\)\..*'):
            label = tf.strings.regex_replace(filename, r'.*\((\d+)\)\..*', r'\1')
        else:
            label = tf.strings.regex_replace(filename, r'.*?(\d+)\..*', r'\1')
        return tf.strings.to_number(label, out_type=tf.int32)

    def load_and_preprocess_image(self, file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [32, 32])
        image = image / 255.0
        return image

    def create_dataset(self, image_type):
        if image_type not in ['FAKE', 'REAL']:
            raise ValueError("image_type must be either 'FAKE' or 'REAL'")
        
        pattern = os.path.join(self.root_dir, '*', image_type, '*.jpg')
        files = tf.data.Dataset.list_files(pattern, shuffle=True)
        
        # Add error handling for empty datasets
        if tf.data.experimental.cardinality(files) == 0:
            raise ValueError(f"No files found matching pattern: {pattern}")
        
        dataset = files.map(
            lambda x: (self.load_and_preprocess_image(x), self.get_label_from_filename(x)), 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset

    def split_dataset(self, dataset, train_split=0.7, val_split=0.15, test_split=0.15):
        # Validate splits
        total = train_split + val_split + test_split
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Splits must sum to 1.0, got {total}")
        
        # Efficient splitting without double iteration
        dataset = dataset.cache()  # Cache before splitting to avoid re-reading
        dataset_size = dataset.reduce(0, lambda x, _: x + 1).numpy()
        
        if dataset_size == 0:
            raise ValueError("Dataset is empty")
        
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        
        train_dataset = dataset.take(train_size)
        remaining = dataset.skip(train_size)
        val_dataset = remaining.take(val_size)
        test_dataset = remaining.skip(val_size)
        
        return train_dataset, val_dataset, test_dataset
