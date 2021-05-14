from __future__ import absolute_import

import tensorflow as tf
from tensorflow import keras

from attacker.constants import *
from data_preprocessing.mnist import MnistPreprocessing
from utility.filters.filter_advs import *
from utility.statistics import *

logger = MyLogger.getLog()


class L_BFGS:
    def __init__(self, classifier, epsilon=0.5, target=7, num_iters=20, weight=0.1):
        """

        :param classifier: target cnn model
        :param epsilon: learning rate
        :param target: target attack
        :param num_iters: number of iterations
        :param weight: balance between target and L2
        """
        self.epsilon = epsilon
        self.target = target
        self.weight = weight
        self.classifier = classifier
        self.target_label = keras.utils.to_categorical(target, MNIST_NUM_CLASSES, dtype='double')
        self.num_iters = num_iters

    @staticmethod
    def loss_func(x_star, origin, model, target):
        return tf.keras.losses.mean_squared_error(x_star, origin) + \
               tf.keras.losses.categorical_crossentropy(model.predict(np.array([x_star]))[0], target)

    @staticmethod
    def create_adversarial_pattern_two_losses(gen_image, input_image, target_label, pretrained_model, weight):
        with tf.GradientTape() as tape:
            tape.watch(gen_image)
            prediction = pretrained_model(gen_image)
            final_loss = (1 - weight) * tf.keras.losses.mean_squared_error(gen_image,
                                                                           input_image) + weight * tf.keras.losses.categorical_crossentropy(
                prediction[0], target_label)
        gradient = tape.gradient(final_loss, gen_image)
        return gradient

    def create_adv_single_image(self, image):
        gen_img = np.array([image])
        image = np.array([image])
        for i in range(self.num_iters):
            grad = self.create_adversarial_pattern_two_losses(tf.convert_to_tensor(gen_img, dtype='double'), image,
                                                              self.target_label, self.classifier, self.weight)
            gen_img -= self.epsilon * grad
            gen_img = np.clip(gen_img, 0, 1)
        return gen_img

    def create_adversaries(self, images):
        result = [self.create_adv_single_image(image) for image in images]
        return np.array(result)


if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 1000
    TARGET = 7

    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    classifier = keras.models.load_model(ATTACKED_CNN_MODEL)

    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

    pre_mnist = MnistPreprocessing(trainX, trainY, testX, testY, START_SEED, END_SEED, TARGET)
    trainX, trainY, testX, testY = pre_mnist.preprocess_data()

    logger.debug('Creating adversarial examples: ')
    lbfgs = L_BFGS(classifier=classifier, target=TARGET)
    result_imgs = lbfgs.create_adversaries(trainX)
    result_origin_imgs, result_origin_confidients, result_gen_imgs, result_gen_confidents = filter_advs(classifier,
                                                                                                        trainX,
                                                                                                        result_imgs,
                                                                                                        TARGET)

    logger.debug('L-BFGS done')
