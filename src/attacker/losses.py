import numpy as np
import tensorflow as tf
from tensorflow import keras

from attacker.constants import *


class AE_LOSSES:
    """
    Provide some loss functions for autoencoder attacker
    """
    CROSS_ENTROPY = 'identity'
    RE_RANK = 're_rank'

    @staticmethod
    def cross_entropy_loss(classifier, target_vector, epsilon):
        """

        :param target_vector:
        :type target_vector:
        :param classifier: classification model
        :param target_label: target_label in one-hot coding
        :param epsilon: balance between target and L2 distance
        :return: loss
        """

        def loss(origin_image, generated_image):
            batch_size = origin_image.shape[0]
            target_vectors = np.repeat(np.array([target_vector]), batch_size, axis=0)
            return (1 - epsilon) * keras.losses.mean_squared_error(
                tf.reshape(origin_image, (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL)),
                tf.reshape(generated_image, (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL))) + \
                   epsilon * keras.losses.categorical_crossentropy(classifier(generated_image)[0], target_vectors)

        return loss

    @staticmethod
    def re_rank_loss(classifier, target_vector, weight, alpha=1.5):
        """

        :param classifier: classification model
        :param target_label: target_label in one-hot coding
        :param weight: balance between target and L2 distance
        :param alpha: // TO_DO
        :return: loss
        """
        target_class = tf.argmax(target_vector)

        def loss(true_image, generated_image):
            re_rank = classifier(true_image).numpy()
            predicted_class = [np.argmax(re_rank_i) for re_rank_i in re_rank]

            for index, predicted_class_i in enumerate(predicted_class):
                re_rank[index, target_class] = alpha * re_rank[index, predicted_class_i]

            re_rank = np.array([(re_rank_i - np.mean(re_rank_i)) / np.std(re_rank_i) for re_rank_i in re_rank])

            batch_size = true_image.shape[0]
            true_image1 = tf.reshape(true_image, (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL))
            generated_image1 = tf.reshape(generated_image,
                                          (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL))

            # print(a)
            return (1 - weight) * tf.keras.losses.mean_squared_error(true_image1, generated_image1) + \
                   weight * tf.keras.losses.mean_squared_error(classifier(generated_image), re_rank)

        return loss

    @staticmethod
    def border_loss(model, target_vector, origin_images, epsilon, shape=(28, 28, 1)):
        def loss(combined_labels, generated_borders):
            borders = combined_labels[:, 0]
            borders = tf.reshape(borders, (borders.shape[0], MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL))
            internal_images = combined_labels[:, 1]
            internal_images = tf.reshape(internal_images,
                                         (internal_images.shape[0], MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL))
            batch_size = borders.shape[0]
            target_vectors = np.repeat(np.array([target_vector]), borders.shape[0], axis=0)

            generated_borders = generated_borders * borders

            border_pixels = combined_labels[:, 2]
            border_pixels = tf.reshape(border_pixels, (
                border_pixels.shape[0], MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)) * borders

            return (1 - epsilon) * keras.losses.mean_squared_error(
                tf.reshape(border_pixels, (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL)),
                tf.reshape(generated_borders, (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL))) + \
                   epsilon * keras.losses.categorical_crossentropy(target_vectors,
                                                                   model(internal_images + generated_borders))

        return loss
