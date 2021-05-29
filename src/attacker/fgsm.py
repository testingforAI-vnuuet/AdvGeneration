from __future__ import absolute_import

import time

from attacker.constants import *
from utility.filters.filter_advs import *

logger = MyLogger.getLog()


class FGSM:
    def __init__(self, trainX, trainY, origin_label, target_label, classifier, weight, num_images, classifier_name='noname'):
        """

        :param classifier: target cnn model
        :param epsilon: epsilon
        :param target: target class
        """

        self.weight = weight
        self.classifier = classifier
        self.origin_label = origin_label
        self.target_label = target_label
        self.num_images = num_images

        self.origin_images = trainX[:self.num_images]



        if self.origin_label is None:
            self.origin_label = 'all'

        self.target_vector = keras.utils.to_categorical(self.target_label, num_classes=MNIST_NUM_CLASSES).reshape(
            (1, MNIST_NUM_CLASSES))

        self.file_shared_name = self.method_name + '_' + classifier_name + f'_{self.origin_label}_{self.target_label}' + 'weight=' + str(
            self.weight).replace('.', ',') + '_' + str(self.num_images)

        self.method_name = 'fgsm'
        self.adv_result = None
        self.origin_adv_result = None
        self.end_time = None
        self.adv_result_file_path = self.file_shared_name + '_adv_result' + '.npy'
        self.origin_adv_result_file_path = self.file_shared_name + '_origin_adv_result' + '.npy'
        self.start_time = time.time()
        self.end_time = None



    @staticmethod
    def create_adversarial_pattern(input_image, target_label, pretrained_model, get_sign=True):
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
        result = [img - self.weight * self.create_adversarial_pattern(img, self.target_label, self.classifier)[0] for
                  img in images]
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
    fgsm = FGSM(classifier=classifier)
    result_imgs = fgsm.create_adversaries(trainX)
    result_origin_imgs, result_origin_confidients, result_gen_imgs, result_gen_confidents = filter_advs(classifier,
                                                                                                        trainX,
                                                                                                        result_imgs,
                                                                                                        TARGET)

    logger.debug('FGSM done')
