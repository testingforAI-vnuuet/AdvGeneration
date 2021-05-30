from __future__ import absolute_import

import os
import time

from attacker.constants import *
from attacker.mnist_utils import reject_outliers
from utility.filters.filter_advs import *
from utility.statistics import filter_candidate_adv

logger = MyLogger.getLog()


class FGSM:
    def __init__(self, trainX, trainY, origin_label, target_label, classifier, weight, num_images,
                 classifier_name='noname'):
        """

        :param classifier: target cnn model
        :param epsilon: epsilon
        :param target: target class
        """

        self.method_name = 'fgsm'
        self.weight = weight
        self.classifier = classifier
        self.origin_label = origin_label
        self.target_label = target_label
        self.num_images = num_images

        self.origin_images = trainX[:self.num_images]
        self.origin_labels = trainY[:self.num_images]

        if self.origin_label is None:
            self.origin_label = 'all'

        self.target_vector = keras.utils.to_categorical(self.target_label, num_classes=MNIST_NUM_CLASSES).reshape(
            (1, MNIST_NUM_CLASSES))

        self.file_shared_name = self.method_name + '_' + classifier_name + f'_{self.origin_label}_{self.target_label}' + 'weight=' + str(
            self.weight).replace('.', ',') + '_' + str(self.num_images)

        self.adv_result = None
        self.origin_adv_result = None
        self.end_time = None
        self.adv_result_file_path = self.file_shared_name + '_adv_result' + '.npy'
        self.origin_adv_result_file_path = self.file_shared_name + '_origin_adv_result' + '.npy'
        self.start_time = time.time()
        self.end_time = None

    @staticmethod
    def create_adversarial_pattern(input_image, target_vector, pretrained_model, get_sign=True):
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
            loss = loss_object(target_vector, prediction)
        gradient = tape.gradient(loss, input)
        sign = tf.sign(gradient) if get_sign is True else gradient
        return sign

    def attack(self):
        generated_candidates = [
            img - self.weight * self.create_adversarial_pattern(img, self.target_vector, self.classifier)[0] for
            img in self.origin_images]

        generated_candidates = np.array(generated_candidates)
        self.adv_result, _, self.origin_adv_result, _ = filter_candidate_adv(self.origin_images,
                                                                             generated_candidates,
                                                                             self.target_label,
                                                                             cnn_model=self.classifier)
        self.end_time = time.time()

    def export_result(self):
        result = '<=========='
        result += '\norigin=' + str(self.origin_label) + ',target=' + str(self.target_label)
        result += '\n\tweight=' + str(self.weight)
        result += '\n\t#adv=' + str(self.adv_result.shape[0])
        l0 = np.array([compute_l0_V2(gen, test) for gen, test in zip(self.adv_result, self.origin_adv_result)])
        l0 = reject_outliers(l0)

        if l0.shape[0] != 0:
            result += '\n\tl0=' + str(min(l0)) + '/' + str(max(l0)) + '/' + str(np.average(l0))
        else:
            result += '\n\tl0=None'

        l2 = np.array([compute_l2_V2(gen, test) for gen, test in zip(self.adv_result, self.origin_adv_result)])
        l2 = reject_outliers(l2)

        if l2.shape[0] != 0:
            result += '\n\tl2=' + str(round(min(l2), 2)) + '/' + str(round(max(l2), 2)) + '/' + str(
                round(np.average(l2), 2))
        else:
            result += '\n\tl2=None'
        result += '\n\ttime=' + str(self.end_time - self.start_time) + ' s'
        result += '\n==========>\n'
        f = open(os.path.join('result', self.method_name, self.file_shared_name + '.txt', ), 'w')
        f.write(result)
        f.close()


if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 1000
    TARGET = 7

    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    classifier = keras.models.load_model(ATTACKED_CNN_MODEL)

    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

    trainX, trainY = MnistPreprocessing.quick_preprocess_data(trainX, trainY, num_classes=MNIST_NUM_CLASSES,
                                                              rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                              chl=MNIST_IMG_CHL)
    testX, testY = MnistPreprocessing.quick_preprocess_data(testX, testY, num_classes=MNIST_NUM_CLASSES,
                                                            rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                            chl=MNIST_IMG_CHL)

    logger.debug('Evaluating classifier on seed_images: ')
    classifier.evaluate(trainX, trainY)

    logger.debug('Creating adversarial examples: ')

    fgsm = FGSM(trainX=trainX, trainY=trainY, origin_label=None, target_label=TARGET, classifier=classifier, weight=0.1,
                num_images=1000, classifier_name='targetmodel')
    fgsm.attack()
    fgsm.export_result()

    logger.debug('FGSM done')
