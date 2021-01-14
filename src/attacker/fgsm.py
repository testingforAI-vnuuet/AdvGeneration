from __future__ import absolute_import

import tensorflow as tf
from tensorflow import keras

from attacker.constants import *
from data_preprocessing.mnist import mnist_preprocessing
from utility.statistics import *

logger = MyLogger.getLog()


class FGSM:
    def __init__(self, epsilon=0.1, target=7):
        self.alpha = epsilon
        self.target = target
        self.target_label = keras.utils.to_categorical(target, num_classes=MNIST_NUM_CLASSES)

    @staticmethod
    def create_adversarial_pattern(input_image, input_label, pretrained_model):
        """
        compute gradient of pretrained model output respect to input_image
        :param input_image: a input image
        :param input_label: the original label corresponding to the input image
        :param pretrained_model: trained classifier
        :return: gradient sign
        """
        loss_object = keras.losses.CategoricalCrossentropy()
        input = input_image if tf.is_tensor(input_image) else tf.convert_to_tensor(input_image)
        with tf.GradientTape() as tape:
            tape.watch(input)
            prediction = pretrained_model(input)
            loss = loss_object(input, prediction)
        gradient = tape.gradient(loss, input)
        sign = tf.sign(gradient)
        return sign

    def create_adversaries(self, images, label):
        result = []


if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 1000
    TARGET = 7

    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    classifier = keras.models.load_model(ATTACKED_CNN_MODEL)

    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

    pre_mnist = mnist_preprocessing(trainX, trainY, testX, testY, START_SEED, END_SEED, TARGET)
    trainX, trainY, testX, testY = pre_mnist.get_preprocess_data()

    logger.log('Evaluating classifier on seed_images: ')
    classifier.evaluate(trainX, trainY)

    logger.log('Creating adversarial examples: ')
    fgsm = FGSM()
    result = fgsm.create_adversaries(trainX, trainY)

