import tensorflow as tf
from tensorflow.keras import backend as K

from typing import Tuple


def base_model(input_shape: Tuple[int, int, int], output_dim: int = 3, num_layers: int = 4, num_channels: int = 256):
    """
    Simple 4-layer MLP (3-hidden layers + 1 output layer) using conv layers
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    if K.image_data_format() == "channels_first":
        x = inputs = tf.keras.layers.Input(shape=(input_shape[-1], input_shape[0], input_shape[1]))
    else:
        x = inputs = tf.keras.layers.Input(shape=input_shape)

    for _ in range(num_layers - 1):
        x = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='valid', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)

    x = tf.keras.layers.Conv2D(output_dim, kernel_size=1, padding='valid', activation='sigmoid')(x)
    return tf.keras.Model(inputs, x, name='base_nerf_model')
