import tensorflow as tf
from tensorflow.keras import backend as K


def base_model(input_shape, output_dim=3):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    if K.image_data_format() == "channels_first":
        x = inputs = tf.keras.layers.Input(shape=(input_shape[-1], input_shape[0], input_shape[1]))
    else:
        x = inputs = tf.keras.layers.Input(shape=input_shape)

    for _ in range(3):
        x = tf.keras.layers.Conv2D(256, kernel_size=1, padding='valid', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(output_dim, kernel_size=1, padding='valid', activation='sigmoid')(x)
    return tf.keras.Model(inputs, x, name='base_nerf_model')