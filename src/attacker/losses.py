import tensorflow as tf
from tensorflow import keras
import numpy as np


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
