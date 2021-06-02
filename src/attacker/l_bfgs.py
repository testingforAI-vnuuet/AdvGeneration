from __future__ import absolute_import

import os
import time

from attacker.constants import *
from attacker.mnist_utils import reject_outliers
from utility.filters.filter_advs import *
from utility.statistics import *

logger = MyLogger.getLog()


class L_BFGS:
    def __init__(self, trainX, trainY, origin_label, target_label, classifier, weight, num_images,
                 classifier_name='noname', num_iter=10):
        """

        :param trainX: origin training set
        :type trainX: np.ndarray
        :param trainY: origin label set
        :type trainY: np.ndarray
        :param origin_label: origin label
        :type origin_label: int or None. Default: None
        :param target_label: target label
        :type target_label: int
        :param classifier: targeted DNN
        :type classifier: keras.models.Model
        :param weight: weight
        :type weight: float
        :param num_images: num of seed images
        :type num_images: int
        :param classifier_name: target DNN name
        :type classifier_name: str
        :param num_iter: num of iterations
        :type num_iter: int
        """

        self.weight = weight
        self.target_label = target_label
        self.origin_label = origin_label
        self.method_name = 'lbfgs'
        if self.origin_label is None:
            self.origin_label = 'all'

        self.classifier = classifier
        self.target_vector = keras.utils.to_categorical(self.target_label, num_classes=MNIST_NUM_CLASSES).reshape(
            (1, MNIST_NUM_CLASSES))
        self.num_iter = num_iter
        self.num_images = num_images
        self.origin_images = trainX[:self.num_images]
        self.origin_labels = trainY[:self.num_images]

        self.file_shared_name = self.method_name + '_' + classifier_name + f'_{self.origin_label}_{self.target_label}' + 'weight=' + str(
            self.weight).replace('.', ',') + '_' + str(self.num_images)

        self.adv_result = None
        self.origin_adv_result = None
        self.end_time = None
        self.adv_result_file_path = self.file_shared_name + '_adv_result' + '.npy'
        self.origin_adv_result_file_path = self.file_shared_name + '_origin_adv_result' + '.npy'
        self.lr = 0.01
        self.start_time = time.time()
        self.end_time = None

    @staticmethod
    def loss_func(x_star, origin, model, target):
        """
        return loss function for L-BFGS attacker
        :param x_star: initial adv
        :type x_star: np.ndarray
        :param origin: an origin image
        :type origin: np.ndarray
        :param model: targeted DNN
        :type model: keras.models.Model
        :param target: target label
        :type target: int
        :return: loss with 2 terms including perception and attack ability
        :rtype: float
        """
        return tf.keras.losses.mean_squared_error(x_star, origin) + \
               tf.keras.losses.categorical_crossentropy(model.predict(np.array([x_star]))[0], target)

    @staticmethod
    def create_adversarial_pattern_two_losses(gen_image, input_image, target_vector, pretrained_model, weight):
        with tf.GradientTape() as tape:
            tape.watch(gen_image)
            prediction = pretrained_model(gen_image)
            final_loss = weight * tf.keras.losses.mean_squared_error(gen_image,
                                                                     input_image) + tf.keras.losses.categorical_crossentropy(
                prediction, target_vector)
        gradient = tape.gradient(final_loss, gen_image)
        return gradient

    def create_adv_single_image(self, image, index):
        """
        creating adv for a single image from original set
        :param index: index of origin image for logging
        :type index: int
        :param image: original image
        :type image: np.ndarray
        :return: generated candidate
        :rtype: np.ndarray
        """
        logger.debug(f"LBFGS: getting candidate for an image: {index}")
        gen_img = np.array([image])
        image = np.array([image])
        for i in range(self.num_iter):
            grad = self.create_adversarial_pattern_two_losses(tf.convert_to_tensor(gen_img, dtype='double'), image,
                                                              self.target_vector, self.classifier, self.weight)
            gen_img -= self.lr * grad
            gen_img = np.clip(gen_img, 0, 1)
        return gen_img[0]

    def attack(self):
        generated_candidates = [self.create_adv_single_image(image, index) for index, image in
                                enumerate(self.origin_images)]
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
        logger.debug('[result] exporting result DONE!')
        abs_path = os.path.abspath(os.path.join('result', self.method_name, self.file_shared_name + '.txt'))
        logger.debug(f'[result] view result at {abs_path}')


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
    logger.debug('[adv] creating adversarial examples start')
    lbfgs = L_BFGS(trainX=trainX, trainY=trainY, origin_label=None, target_label=TARGET, classifier=classifier,
                   weight=0.0001,
                   num_images=1000, classifier_name='targetmodel', num_iter=20)
    lbfgs.attack()
    lbfgs.export_result()
    logger.debug('[adv] creating adversarial examples DONE')

    logger.debug('L-BFGS done')
