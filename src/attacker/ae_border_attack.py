"""
Created At: 23/03/2021 16:22
"""

import threading
import time

from attacker.autoencoder import *
from attacker.constants import *
from attacker.mnist_utils import *
from utility.statistics import *

tf.config.experimental_run_functions_eagerly(True)

logger = MyLogger.getLog()

pretrained_model_name = ['Alexnet', 'Lenet', 'vgg13', 'vgg16']


def combined_function(set1, set2, set3):
    return np.array([list(combined) for combined in zip(set1, set2, set3)])


class AutoencoderBorder:
    def __init__(self, origin_label, trainX, trainY, classifier, target_position=2, classifier_name='noname'):
        """

        :param origin_label:
        :type origin_label:
        :param trainX:
        :type trainX:
        :param trainY:
        :type trainY:
        :param classifier:
        :type classifier:
        :param target_position:
        :type target_position:
        """
        self.start_time = time.time()
        self.end_time = None
        self.origin_label = origin_label
        self.trainX = trainX
        self.trainY = trainY
        self.classifier = classifier
        self.target_position = target_position

        logger.debug('init attacking: origin_label = {origin_label}'.format(origin_label=self.origin_label))

        self.origin_images, self.origin_labels = filter_by_label(self.origin_label, self.trainX, self.trainY)

        self.origin_images = np.array(self.origin_images[:2000])
        self.origin_labels = np.array(self.origin_labels[:2000])

        logger.debug('shape of origin_images: {shape}'.format(shape=self.origin_images.shape))
        logger.debug('shape of origin_labels: {shape}'.format(shape=self.origin_labels.shape))

        self.target_label = label_ranking(self.origin_images, self.classifier)[-1 * self.target_position]
        self.target_vector = tf.keras.utils.to_categorical(self.target_label, MNIST_NUM_CLASSES, dtype='float32')
        logger.debug('target_label: {target_label}'.format(target_label=self.target_label))

        # logger.debug('ranking sample')
        # self.origin_images, self.origin_labels = ranking_sample(self.origin_images, self.origin_labels,
        #                                                         self.origin_label, self.target_label, self.classifier)
        # logger.debug('shape of ranking sample: {shape}'.format(shape=self.origin_images.shape))
        # logger.debug('ranking sample DONE!')

        self.border_origin_images = get_border(self.origin_images)
        logger.debug('border_origin_image shape: {shape}'.format(shape=self.border_origin_images.shape))
        self.internal_origin_images = get_internal_images(self.origin_images, border_images=self.border_origin_images)
        logger.debug('internal_origin_image shape: {shape}'.format(shape=self.internal_origin_images.shape))
        self.file_shared_name = classifier_name + 'border_{origin_label}_{target_label}'.format(
            origin_label=self.origin_label,
            target_label=self.target_label)
        self.autoencoder_file_name = 'autoencoder_' + self.file_shared_name + '.h5'
        self.result_file_name = 'result_' + self.file_shared_name + '.txt'

        self.result_folder = './result'

        logger.debug('combining target_labels for autoencoder training')

        self.combined_labels = combined_function(self.border_origin_images, self.internal_origin_images, self.origin_images)
        self.autoencoder = None
        self.generated_borders = None
        self.generated_candidates = None
        self.adv_result = None
        self.origin_adv_result = None
        logger.debug('init attacking DONE!')

    def autoencoder_attack(self, loss, epsilon=0.5):
        ae_trainee = MnistAutoEncoder()

        if os.path.isfile(SAVED_ATTACKER_PATH + '/' + self.autoencoder_file_name):
            logger.debug(
                'found pre-trained autoencoder for: origin_label = {origin_label}, target_label = {target_label}'.format(
                    origin_label=self.origin_label, target_label=self.target_label))

            self.autoencoder = tf.keras.models.load_model(SAVED_ATTACKER_PATH + '/' + self.autoencoder_file_name,
                                                          compile=False)
            adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.autoencoder.compile(optimizer=adam,
                                     loss=loss(self.classifier, self.target_vector, self.origin_images, epsilon))
            self.end_time = time.time()

            return

        else:
            logger.debug(
                'not found pre-trained autoencoder for: origin_label = {origin_label}, target_label = {target_label}'.format(
                    origin_label=self.origin_label, target_label=self.target_label))
            logger.debug('training autoencoder for: origin_label={origin_label}, target_label={target_label}'.format(
                origin_label=self.origin_label, target_label=self.target_label))

            self.autoencoder = ae_trainee.get_architecture()
            adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.autoencoder.compile(optimizer=adam,
                                     loss=loss(self.classifier, self.target_vector, self.origin_images, epsilon))
            # self.autoencoder.compile(optimizer=adam, loss=tf.keras.losses.binary_crossentropy)
            early_stopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='min')
            model_checkpoint = ModelCheckpoint(SAVED_ATTACKER_PATH + '/' + self.autoencoder_file_name,
                                               save_best_only=True, monitor='loss',
                                               mode='min')

            self.autoencoder.fit(self.origin_images, self.combined_labels, epochs=200, batch_size=512,
                                 callbacks=[early_stopping, model_checkpoint], verbose=1)
            logger.debug('training autoencoder DONE!')

    def get_border_and_adv(self):

        # self.generated_borders = []
        # self.generated_candidates = []
        logger.debug('getting advs')
        candidate_generated_borders = self.autoencoder.predict(self.origin_images) * self.border_origin_images

        self.generated_candidates = np.clip(candidate_generated_borders + self.internal_origin_images, 0, 1)
        self.generated_borders = np.array(candidate_generated_borders)
        self.adv_result, _, self.origin_adv_result, _ = filter_candidate_adv(self.origin_images,
                                                                             self.generated_candidates,
                                                                             self.target_label,
                                                                             cnn_model=self.classifier)
        self.end_time = time.time()
        logger.debug('get advs DONE!')

    def export_result(self):
        result = '<=========='
        result += '\norigin=' + str(self.origin_label) + ',target=' + str(self.target_label) + '\n'
        result += '\n\t#adv=' + str(self.adv_result.shape[0])
        l0 = np.array([L0(gen, test) for gen, test in zip(self.adv_result, self.origin_adv_result)])
        l0 = reject_outliers(l0)

        if l0.shape[0] != 0:
            result += '\n\tl0=' + str(min(l0)) + '/' + str(max(l0)) + '/' + str(np.average(l0))
        else:
            result += '\n\tl0=None'

        l2 = np.array([L2(gen, test) for gen, test in zip(self.adv_result, self.origin_adv_result)])
        l2 = reject_outliers(l2)

        if l2.shape[0] != 0:
            result += '\n\tl2=' + str(round(min(l2), 2)) + '/' + str(round(max(l2), 2)) + '/' + str(
                round(np.average(l2), 2))
        else:
            result += '\n\tl2=None'
        result += '\n\ttime=' + str(self.end_time - self.start_time) + ' s'
        result += '\n==========>\n'
        return result, self.end_time - self.start_time


def run_thread(classifier_name, trainX, trainY):
    logger.debug("\n=======================================================")
    logger.debug('processing model: ' + classifier_name)
    cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    result_txt = classifier_name + '\n'
    # AE_LOSS = AE_LOSSES.border_loss
    for origin_label in range(0, 10):
        exe_time_sum = 0
        for target_position in range(2, 11):
            attacker = AutoencoderBorder(origin_label, np.array(trainX), np.array(trainY), cnn_model,
                                         target_position=target_position, classifier_name=classifier_name)
            attacker.autoencoder_attack(loss=AE_LOSSES.border_loss)
            attacker.get_border_and_adv()
            res_txt, exe_time = attacker.export_result()
            result_txt += res_txt
            exe_time_sum += exe_time
        f = open('./result/' + classifier_name + str(origin_label) + '.txt', 'w')
        result_txt += '\n average_time = ' + str(exe_time_sum / 9.) + '\n'
        f.write(result_txt)
        f.close()
        logger.debug('processing model: ' + classifier_name + ' DONE!')
        logger.debug("=======================++++============================")


class MyThread(threading.Thread):
    def __init__(self, classifier_name, trainX, trainY):
        super(MyThread, self).__init__()
        self.classifier_name = classifier_name
        self.trainX = trainX
        self.trainY = trainY

    def run(self):
        run_thread(self.classifier_name, self.trainX, self.trainY)


if __name__ == '__main__':
    # target_position = 2

    logger.debug('pre-processing data')
    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

    trainX, trainY = MnistPreprocessing.quick_preprocess_data(trainX, trainY, num_classes=MNIST_NUM_CLASSES,
                                                              rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                              chl=MNIST_IMG_CHL)
    testX, testY = MnistPreprocessing.quick_preprocess_data(testX, testY, num_classes=MNIST_NUM_CLASSES,
                                                            rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                            chl=MNIST_IMG_CHL)
    logger.debug('pre-processing data DONE !')

    logger.debug('robustness testing start')

    # AE_LOSS = AE_LOSSES.border_loss
    # for classifier_name in pretrained_model_name:
    #     logger.debug("\n=======================================================")
    #     logger.debug('processing model: ' + classifier_name)
    #     cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    #     result_txt = classifier_name + '\n'
    #     for origin_label in range(0, 2):
    #         exe_time_sum = 0
    #         for target_position in range(2, 4):
    #             attacker = AutoencoderBorder(origin_label, np.array(trainX), np.array(trainY), cnn_model,
    #                                          target_position=target_position, classifier_name=classifier_name)
    #             attacker.autoencoder_attack(loss=AE_LOSS)
    #             attacker.get_border_and_adv()
    #             res_txt, exe_time = attacker.export_result()
    #             result_txt += res_txt
    #             exe_time_sum += exe_time
    #         f = open('./result/' + classifier_name + '.txt', 'w')
    #         result_txt += '\n average_time = ' + str(exe_time_sum / 9. + '\n')
    #         f.write(result_txt)
    #         f.close()
    #         logger.debug('processing model: ' + classifier_name + ' DONE!')
    #         logger.debug("=======================++++============================")

    logger.debug('starting multi-thread')
    thread1 = MyThread(pretrained_model_name[2], trainX, trainY)
    thread2 = MyThread(pretrained_model_name[3], trainX, trainY)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    logger.debug('Exiting Main Thread')
    logger.debug('robustness testing DONE !')
