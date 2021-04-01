import numpy as np
import tensorflow as tf
from tensorflow import keras


class AE_LOSSES:
    """
    Provide some loss functions for autoencoder attacker
    """
    CROSS_ENTROPY = 'identity'
    RE_RANK = 're_rank'

    @staticmethod
    def cross_entropy_loss(classifier, target_label, epsilon):
        """

        :param classifier: classification model
        :param target_label: target_label in one-hot coding
        :param epsilon: balance between target and L2 distance
        :return: loss
        """

        def loss(origin_image, generated_image):
            return (1 - epsilon) * keras.losses.mean_squared_error(origin_image, generated_image) + \
                   epsilon * keras.losses.categorical_crossentropy(classifier(generated_image)[0], target_label)

        return loss

    @staticmethod
    def re_rank_loss(classifier, target_label, weight, alpha):
        """

        :param classifier: classification model
        :param target_label: target_label in one-hot coding
        :param weight: balance between target and L2 distance
        :param alpha: // TO_DO
        :return: loss
        """
        target_class = tf.argmax(target_label)

        def loss(origin_image, generated_image):
            re_rank = classifier(origin_image)[0].numpy()
            predicted_class = np.argmax(re_rank)
            re_rank[target_class] = alpha * re_rank[predicted_class]
            re_rank = (re_rank - np.mean(re_rank)) / np.std(re_rank)
            return (1 - weight) * tf.keras.losses.mean_squared_error(origin_image, generated_image) + \
                   weight * tf.keras.losses.mean_squared_error(classifier(generated_image)[0], re_rank)

        return loss

    @staticmethod
    def border_loss(model, target_vector, origin_images, epsilon, shape=(28, 28, 1)):
        def loss(combined_labels, generated_borders):
            borders = combined_labels[:, 0]
            borders = tf.reshape(borders, (borders.shape[0], 28, 28, 1))
            internal_images = combined_labels[:, 1]
            internal_images = tf.reshape(internal_images, (internal_images.shape[0], 28, 28, 1))
            a = borders.shape[0]
            target_vectors = np.repeat(np.array([target_vector]), borders.shape[0], axis=0)

            generated_borders = generated_borders * borders

            return (1 - epsilon) * keras.losses.binary_crossentropy(tf.reshape(borders, (a, 784)),
                                                                    tf.reshape(generated_borders, (a, 784))) + \
                   epsilon * keras.losses.categorical_crossentropy(target_vectors,
                                                                   model(internal_images + generated_borders))
        return loss
