import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, Input, MultiHeadAttention, LayerNormalization, Add, Layer
from tensorflow.keras.models import Model
import yaml
import os

class ModelCreator:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['models']

    def create_model(self, model_type):
        if model_type not in self.config:
            raise ValueError(f"Unsupported model type: {model_type}")
        config = self.config[model_type]
        if model_type == 'CNN':
            return self.create_cnn(config)
        elif model_type == 'VisionTransformer':
            return self.create_vision_transformer(config)
        elif model_type == 'MLP':
            return self.create_mlp(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def create_cnn(self, config):
        input_shape = config['input_shape']
        num_classes = config['num_classes']
        conv_layers = config['conv_layers']
        batch_norm = config['batch_norm']
        dropout_rate = config['dropout_rate']
        dense_layers = config['dense_layers']
        final_activation = config['final_activation']
        pooling = config.get('pooling', {})
        padding = config.get('padding', 'valid')

        # He initialization for convolutional and dense layers
        he_initializer = tf.keras.initializers.HeNormal()
        # Zero initialization for final classifier
        zero_initializer = tf.keras.initializers.Zeros()

        inputs = Input(shape=input_shape)
        x = inputs

        for layer in conv_layers:
            # Conv → BN → ReLU pattern with He initialization and same padding
            x = Conv2D(
                filters=layer['filters'], 
                kernel_size=layer['kernel_size'], 
                activation=None,  # No activation here
                padding=padding,
                kernel_initializer=he_initializer
            )(x)
            
            if batch_norm:
                x = BatchNormalization()(x)
            
            # Apply activation after batch norm
            x = tf.keras.layers.Activation(layer['activation'])(x)
            
            # Add pooling layer if specified
            if pooling.get('enabled', False):
                pool_size = pooling.get('pool_size', [2, 2])
                pool_type = pooling.get('type', 'max')
                
                if pool_type == 'max':
                    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)
                elif pool_type == 'average':
                    x = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)

        x = Flatten()(x)
        x = Dropout(dropout_rate)(x)

        for layer in dense_layers:
            x = Dense(
                units=layer['units'], 
                activation=layer['activation'],
                kernel_initializer=he_initializer
            )(x)
            x = Dropout(dropout_rate)(x)

        # Final classifier with zero initialization
        outputs = Dense(
            units=num_classes, 
            activation=final_activation,
            kernel_initializer=zero_initializer,
            bias_initializer=zero_initializer
        )(x)

        return Model(inputs=inputs, outputs=outputs)

    def create_vision_transformer(self, config):
        input_shape = config['input_shape']
        num_classes = config['num_classes']
        patch_size = config['patch_size']
        num_patches = config['num_patches']
        projection_dim = config['projection_dim']
        transformer_layers = config['transformer_layers']
        dense_units = config['dense_units']
        dropout_rate = config['dropout_rate']
        final_activation = config['final_activation']

        class Patches(Layer):
            def __init__(self, patch_size):
                super(Patches, self).__init__()
                self.patch_size = patch_size

            def call(self, images):
                batch_size = tf.shape(images)[0]
                patches = tf.image.extract_patches(
                    images=images,
                    sizes=[1, self.patch_size[0], self.patch_size[1], 1],
                    strides=[1, self.patch_size[0], self.patch_size[1], 1],
                    rates=[1, 1, 1, 1],
                    padding="VALID",
                )
                patch_dims = patches.shape[-1]
                patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
                return patches

        class PatchEncoder(Layer):
            def __init__(self, num_patches, projection_dim):
                super(PatchEncoder, self).__init__()
                self.num_patches = num_patches
                self.projection = Dense(units=projection_dim)
                self.position_embedding = tf.Variable(
                    tf.random.normal([1, num_patches, projection_dim]), trainable=True
                )

            def call(self, patch):
                positions = tf.range(start=0, limit=self.num_patches, delta=1)
                encoded = self.projection(patch) + self.position_embedding[:, positions, :]
                return encoded

        inputs = Input(shape=input_shape)
        x = Patches(patch_size)(inputs)
        x = PatchEncoder(num_patches, projection_dim)(x)

        for layer in transformer_layers:
            x = LayerNormalization(epsilon=1e-6)(x)
            x1 = MultiHeadAttention(
                num_heads=layer['num_heads'], key_dim=projection_dim, dropout=layer['dropout_rate']
            )(x, x)
            x = Add()([x, x1])
            x = LayerNormalization(epsilon=1e-6)(x)
            x1 = tf.keras.Sequential(
                [
                    Dense(units=layer['mlp_units'][0], activation=tf.nn.gelu),
                    Dropout(rate=layer['dropout_rate']),
                    Dense(units=projection_dim),
                ]
            )(x)
            x = Add()([x, x1])

        x = LayerNormalization(epsilon=1e-6)(x)
        x = Flatten()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(units=dense_units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(units=num_classes, activation=final_activation)(x)

        return Model(inputs=inputs, outputs=outputs)

    def create_mlp(self, config):
        input_shape = config['input_shape']
        num_classes = config['num_classes']
        flatten = config['flatten']
        dense_layers = config['dense_layers']
        dropout_rate = config['dropout_rate']
        final_activation = config['final_activation']

        inputs = Input(shape=input_shape)
        x = inputs

        if flatten:
            x = Flatten()(x)

        for layer in dense_layers:
            x = Dense(units=layer['units'], activation=layer['activation'])(x)
            x = Dropout(dropout_rate)(x)

        outputs = Dense(units=num_classes, activation=final_activation)(x)

        return Model(inputs=inputs, outputs=outputs)

