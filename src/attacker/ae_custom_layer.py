"""
Created At: 11/06/2021 17:03
"""
import numpy as np
import tensorflow as tf

from utility.constants import float32_type

tf.config.experimental_run_functions_eagerly(True)


class concate_start_to_end(tf.keras.layers.Layer):
    def __init__(self, important_pixels, num_pixels=784):
        super(concate_start_to_end, self).__init__()
        self.important_pixels = np.array(important_pixels)

        self.not_important_pixels = np.array(
            [pixel for pixel in range(num_pixels) if pixel not in self.important_pixels])

        self.num_pixels = num_pixels
        self.batch_size = None

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs):
        generated, origin = inputs
        self.batch_size = generated.shape[0]
        if self.batch_size is None:
            return
        origin_shape = generated.shape

        # flat inputs
        # print(batch_size)
        generated = tf.reshape(tensor=generated, shape=(self.batch_size, -1))
        origin = tf.reshape(tensor=origin, shape=(self.batch_size, -1))

        # create mask
        important_pixels_array = np.zeros(shape=generated.shape, dtype=np.float32)
        # print(important_pixels_array.shape)
        important_pixels_array[:, self.important_pixels] = 1.0

        important_pixels_array = tf.Variable(important_pixels_array, trainable=False, dtype=float32_type)

        not_important_pixel_array = np.zeros(shape=generated.shape, dtype=np.float32)
        not_important_pixel_array[:, self.not_important_pixels] = 1.0
        not_important_pixel_array = tf.Variable(not_important_pixel_array, trainable=False, dtype=float32_type)

        # get result
        result = tf.add(tf.math.multiply(generated, important_pixels_array), tf.multiply(origin, not_important_pixel_array))
        # print(result.shape)
        # print(result.numpy()[:, self.important_pixels])
        result = tf.reshape(tensor=result, shape=origin_shape)


        # print(tf.subtract(result, ))
        return result
