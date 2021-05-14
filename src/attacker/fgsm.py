from __future__ import absolute_import

import tensorflow as tf

from attacker.constants import *
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *
from utility.filter_advs import *
logger = MyLogger.getLog()


class FGSM:
    def __init__(self, classifier, epsilon=0.1, target=7):
        """

        :param classifier: target cnn model
        :param epsilon: epsilon
        :param target: target class
        """

        self.epsilon = epsilon
        self.target = target
        self.classifier = classifier
        self.target_label = keras.utils.to_categorical(target, num_classes=MNIST_NUM_CLASSES).reshape((1, MNIST_NUM_CLASSES))

    @staticmethod
    def create_adversarial_pattern(input_image, target_label, pretrained_model, get_sign = True):
        """
        compute gradient of pretrained model output respect to input_image
        :param input_image: a input image
        :param target_label: the target label corresponding to the input image
        :param pretrained_model: trained classifier
        :return: gradient sign
        """
        loss_object = keras.losses.CategoricalCrossentropy()
        input = [input_image] if tf.is_tensor(input_image) else tf.convert_to_tensor([input_image])
        with tf.GradientTape() as tape:
            tape.watch(input)
            prediction = pretrained_model(input)
            loss = loss_object(target_label, prediction)
        gradient = tape.gradient(loss, input)
        sign = tf.sign(gradient) if get_sign == True else gradient
        return sign

    def create_adversaries(self, images):
        result = [img - self.epsilon * self.create_adversarial_pattern(img, self.target_label, self.classifier)[0] for img in images]
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

    logger.debug('Evaluating classifier on seed_images: ')
    classifier.evaluate(trainX, trainY)

    logger.debug('Creating adversarial examples: ')
    fgsm = FGSM(classifier= classifier, target=TARGET)
    result_imgs = fgsm.create_adversaries(trainX)
    result_origin_imgs, result_origin_confidients, result_gen_imgs, result_gen_confidents = filter_advs(classifier, trainX, result_imgs, TARGET)

    logger.debug('Fgsm done')

