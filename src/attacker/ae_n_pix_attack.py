"""
Created At: 18/03/2021 11:09
"""
import os
import time

import numpy as np

from attacker.autoencoder import *
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *

logger = MyLogger.getLog()

METHOD_NAME = 'AE_NPIX'
AE_PHASE_NAME = 'AE4DNN'
NPIX_PHASE_NAME = 'N-PIX'


class AutoencoderNPix:

    def __init__(self, origin_label, trainX, trainY, classifier, num_pixel=10, target_position=2, num_seed_data=1000):
        """


        :param origin_label: origin label
        :type origin_label: int
        :param trainX: training data for autoencoder
        :type trainX: np.ndarray
        :param trainY: trainin label for autoencoder in one-hot type
        :type trainY: np.ndarray
        :param target_position: position of target label in ranked labels by classifier associated with origin_label
        :type target_position: int
        :param num_seed_data: for n-pix attack
        :type num_seed_data: int
        """
        self.origin_label = origin_label
        self.trainX = trainX
        self.trainY = trainY
        self.target_position = target_position
        self.num_seed_data = num_seed_data
        self.autoencoder_file_name = 'autoencoderNPix_{origin_label}_{target_label}.h5'
        self.classifier = classifier
        self.num_pixel = num_pixel

        # target_label: target label for attacking associated with 'target_position'
        self.target_label = None
        self.target_vector = None

        # lamda: for initiated by the autoencoder;type: np.ndarray
        self.lamda = None

        #  autoencoder phase
        # Sx_for_training_ae: seed data for autoencoder attack;type: np.ndarray
        self.Sx_for_training_ae = None

        # Sy_for_training_ae: label set related to 'Sx_for_training_ae';type: np.ndarray
        self.Sy_for_training_ae = None

        self.autoencoder_adv = None

        self.autoencoder_adv_labels_vector = None

        self.autoencoder_origin_for_adv = None

        self.autoencoder_origin_for_adv_labels_vector = None

        # n-pix phase
        self.Sx_for_training_n_pix = None

        self.Sy_for_training_n_pix = None

        self.n_pix_adv = None

        self.n_pix_adv_labels_vector = None

        self.n_pix_origin_for_adv = None

        self.n_pix_origin_for_adv_labels_vector = None
        # autoencoder model
        self.autoencoder = None

        self.mask = np.zeros(MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL, )
        self.loss_history = []
        self.start_time = time.time()
        self.end_time = None
        self.autoencoder_success_rate = None
        self.result_dir = './result'

    def __init_ae_attack(self):
        self.trainX, self.trainY = filter_by_label(self.origin_label, self.trainX, self.trainY)
        logger.debug('{phase_name}: shape of train data by origin label for autoeencoder: {shape}'.format(
            phase_name=AE_PHASE_NAME,
            shape=self.trainX.shape))
        logger.debug('{phase_name}: shape of train label by origin label for autoencoder: {shape}'.format(
            phase_name=AE_PHASE_NAME,
            shape=self.trainY.shape))

        self.Sx_for_training_ae = np.array(self.trainX[:-1])  # todo
        self.Sy_for_training_ae = np.array(self.trainY[:-1])  # todo

        logger.debug('{phase_name}: shape of data set for autoencoder attack: {shape}'.format(phase_name=AE_PHASE_NAME,
                                                                                              shape=self.Sx_for_training_ae.shape))
        logger.debug('{phase_name}: shape of label set for autoencoder attack: {shape}'.format(phase_name=AE_PHASE_NAME,
                                                                                               shape=self.Sy_for_training_ae.shape))

        self.target_label = label_ranking(self.trainX[:self.num_seed_data], self.classifier)[-1 * self.target_position]
        self.target_vector = keras.utils.to_categorical(self.target_label, MNIST_NUM_CLASSES, dtype='float32')
        logger.debug('{phase_name}: target label: {target}'.format(phase_name=METHOD_NAME, target=self.target_label))

        self.autoencoder_file_name = self.autoencoder_file_name.format(origin_label=self.origin_label,
                                                                       target_label=self.target_label)

    def l2_attack_by_ae(self, loss):

        # filtering by origin_label and finding target_label
        self.__init_ae_attack()
        ae_trainee = MnistAutoEncoder()
        # checking if autoencoder is already trained
        if os.path.isfile(self.autoencoder_file_name):
            logger.debug('{phase_name}: found pre-trained autoencoder model'.format(phase_name=AE_PHASE_NAME))
            self.autoencoder = tf.keras.models.load_model(self.autoencoder_file_name, compile=False)

            self.autoencoder = ae_trainee.train(auto_encoder=self.autoencoder, attacked_classifier=self.classifier,
                                                loss=loss, epochs=0,
                                                batch_size=256, training_set=self.Sx_for_training_ae, epsilon=0.02,
                                                output_model_path=self.autoencoder_file_name,
                                                target_label=self.target_label, is_fit=False)
            return
        else:
            logger.debug('{phase_name}: not found pre-trained autoencoder model'.format(phase_name=AE_PHASE_NAME))
            logger.debug(
                '{phase_name}: training autoencoder start: origin_label={origin_label}, target_label={target_label}'.format(
                    phase_name=AE_PHASE_NAME, origin_label=self.origin_label, target_label=self.target_label))

            architecture = ae_trainee.get_architecture()
            self.autoencoder = ae_trainee.train(auto_encoder=architecture, attacked_classifier=self.classifier,
                                                loss=loss, epochs=400,
                                                batch_size=256, training_set=self.Sx_for_training_ae, epsilon=0.02,
                                                output_model_path=self.autoencoder_file_name,
                                                target_label=self.target_label)

            logger.debug('{phase_name}: training autoencoder done !'.format(phase_name=AE_PHASE_NAME))

    def __init_n_pix_attack(self):
        candidate_adv = self.autoencoder.predict(self.trainX)
        self.autoencoder_adv, self.autoencoder_adv_labels_vector, self.autoencoder_origin_for_adv, self.autoencoder_origin_for_adv_labels_vector = filter_candidate_adv(
            self.trainX, candidate_adv, self.target_label, self.classifier)

        self.autoencoder_success_rate = self.autoencoder_adv.shape[0] / self.trainX.shape[0]

        logger.debug(
            '{phase_name}: success rate on {num_origin_imgs} images (label: {origin_label}): {success_rate}'.format(
                phase_name=METHOD_NAME,
                num_origin_imgs=self.trainX.shape[0],
                origin_label=self.origin_label,
                success_rate=self.autoencoder_success_rate
            ))

        self.Sx_for_training_n_pix = self.autoencoder_origin_for_adv[:self.num_seed_data]
        self.Sy_for_training_n_pix = self.autoencoder_adv_labels_vector[:self.num_seed_data]

        # compute lamda
        self.lamda = compute_difference_two_set(self.autoencoder_adv[:self.num_seed_data], self.Sx_for_training_n_pix)

        logger.debug('{phase_name}: computing mask by ranking pixel'.format(phase_name=NPIX_PHASE_NAME))
        logger.debug('{phase_name}: shape of seed_images set: {shape}'.format(phase_name=NPIX_PHASE_NAME,
                                                                              shape=self.Sx_for_training_n_pix.shape))
        pixels_index = self.__pixel_ranking()
        self.mask = np.zeros(MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL, )
        self.mask[pixels_index] = 1
        self.mask = self.mask.reshape((MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL))

        logger.debug(
            '{phase_name}: shape of the mask: {shape}'.format(phase_name=NPIX_PHASE_NAME, shape=self.mask.shape))

        if int(sum(self.mask.flatten()) != self.num_pixel):
            logger.error('{phase_name}: error - computing mask'.format(phase_name=NPIX_PHASE_NAME))
        else:
            logger.debug('{phase_name}: computing mask done  !'.format(phase_name=NPIX_PHASE_NAME))

    def __gradient_by_target(self, image):
        lamda_tensor = tf.convert_to_tensor([self.lamda])
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        with tf.GradientTape() as tape:
            tape.watch(lamda_tensor)
            img = np.array([image]) + lamda_tensor
            prediction = self.classifier(img)[0]
            loss = loss_object(self.target_vector, prediction)

        return tape.gradient(loss, lamda_tensor)[0]

    def __pixel_ranking(self):
        sum_gradient = np.zeros(shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL))
        for index, image in enumerate(self.Sx_for_training_n_pix):
            sum_gradient += self.__gradient_by_target(image)
        flat_sum = np.array(sum_gradient).reshape(784, )
        return np.argsort(flat_sum)[-1 * self.num_pixel:]

    def l0_attack_by_n_pix(self, iterations):
        self.__init_n_pix_attack()
        # gradient set up
        prev = 0
        curr = 0
        lamda_tensor = tf.convert_to_tensor([self.lamda])

        logger.debug('{phase_name}: finishing optimized lamda'.format(phase_name=NPIX_PHASE_NAME))
        for iteration in range(iterations):
            logger.debug('{phase_name}: iteration number: {index}'.format(phase_name=NPIX_PHASE_NAME, index=iteration))
            lamda_tensor = tf.convert_to_tensor(lamda_tensor)
            with tf.GradientTape() as tape:
                tape.watch(lamda_tensor)
                sum_loss = None

                for index, image in enumerate(self.Sx_for_training_n_pix):
                    img = np.array([image]) + self.mask * lamda_tensor
                    img = tf.experimental.numpy.clip(img, 0, 1)

                    loss_item = tf.keras.losses.categorical_crossentropy(self.target_vector, self.classifier(img)[0])
                    if sum_loss is None:
                        sum_loss = loss_item / self.num_seed_data
                    else:
                        sum_loss += loss_item / self.num_seed_data
            gradient = tape.gradient(sum_loss, lamda_tensor)
            curr = 0.9 * prev + 0.5 * gradient
            lamda_tensor -= curr
            prev = curr
            self.loss_history.append(sum_loss.numpy())

        self.lamda = lamda_tensor.numpy()

        logger.debug('{phase_name}: finding optimized lamda done  !'.format(phase_name=NPIX_PHASE_NAME))

    def __adding_noise(self, data_set: np.ndarray, lamda: np.ndarray):
        result = []
        data_set = np.array(data_set)
        for data_i in data_set:
            img = np.array(data_i + self.mask * self.lamda[0])
            img = np.clip(img, 0, 1)
            result.append(img)
        return np.array(result)

    def export_result(self):
        logger.debug('{phase_name}: exporting results '.format(phase_name=METHOD_NAME))
        self.end_time = time.time()
        name_template = str(self.origin_label) + '_' + str(self.target_label) + '_pix_' + str(self.num_pixel)

        np.save(self.result_dir + '/lamda_' + name_template + '.npy', self.lamda)

        result_text = ''
        for num in [100, 1000, -1]:
            result_num = self.__adding_noise(self.autoencoder_origin_for_adv[:num], self.lamda)
            result_advs, result_adv_labels, result_origin, result_origin_labels = filter_candidate_adv(
                self.autoencoder_origin_for_adv[:num], result_num, self.target_label, self.classifier)
            result_suc_rate = (result_advs.shape[0] / result_num.shape[0]) * 100.0
            if num == 100:
                np.save(self.result_dir + '/ori_' + name_template + '.npy', result_origin)
                np.save(self.result_dir + '/adv_' + name_template + '.npy', result_advs)
            result_text += 'success_rate on ' + str(result_num.shape[0]) + ': ' + str(result_suc_rate) + '\n'

        exe_time = self.end_time - self.start_time
        result_text += '\nexecution time: ' + str(exe_time) + 's'
        result_text += '\nhistory_loss: ' + str(self.loss_history)
        f = open(self.result_dir + '/' + name_template + '.txt', 'w')
        f.write(result_text)
        f.close()
        logger.debug('{phase_name}: exporting results done !'.format(phase_name=METHOD_NAME))


if __name__ == '__main__':

    target_position = 2

    logger.debug('{phase_name}: pre-processing data'.format(phase_name=METHOD_NAME))
    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

    trainX, trainY = MnistPreprocessing.quick_preprocess_data(trainX, trainY, num_classes=MNIST_NUM_CLASSES,
                                                              rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                              chl=MNIST_IMG_CHL)
    testX, testY = MnistPreprocessing.quick_preprocess_data(testX, testY, num_classes=MNIST_NUM_CLASSES,
                                                            rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                            chl=MNIST_IMG_CHL)
    logger.debug('{phase_name}: pre-processing data done !'.format(phase_name=METHOD_NAME))

    logger.debug('{phase_name}: robustness testing start'.format(phase_name=METHOD_NAME))

    for origin_label in range(0, 10):
        for num_pixel in [5, 10, 20]:
            logger.debug(
                '\n\n=============origin: {origin}, n-pix: {pix}==================='.format(origin=origin_label,
                                                                                            pix=num_pixel))
            AE_LOSS = AE_LOSSES.cross_entropy_loss

            cnn_model = keras.models.load_model(CLASSIFIER_PATH + MNIST_CNN_MODEL_WITH_PRE_SOFTMAX_FILE)

            attacker = AutoencoderNPix(origin_label, trainX, trainY, cnn_model, num_pixel=num_pixel,
                                       target_position=target_position)

            attacker.l2_attack_by_ae(AE_LOSS)
            attacker.l0_attack_by_n_pix(iterations=10)
            attacker.export_result()
            logger.debug('==================================================================')

    logger.debug('{phase_name}: robustness testing done  !'.format(phase_name=METHOD_NAME))
