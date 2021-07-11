"""
Created At: 23/03/2021 16:22
"""
import os.path
import threading
import time

from attacker.autoencoder import *
from attacker.constants import *
from attacker.mnist_utils import *
from utility.statistics import *

tf.config.experimental_run_functions_eagerly(True)

logger = MyLogger.getLog()

pretrained_model_name = ['Alexnet', 'Lenet_v2', 'vgg13', 'vgg16']


def combined_function(set1, set2, set3):
    return np.array([list(combined) for combined in zip(set1, set2, set3)])


class AAE_V2:
    def __init__(self, origin_label, trainX, trainY, classifier, weight, target_position=2, classifier_name='noname',
                 step=12,
                 num_images=1000):
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
        self.optimal_epoch = 0
        self.weight = weight
        self.num_images = num_images
        self.method_name = AAE_METHOD_NAME
        self.step = step

        logger.debug('init attacking: origin_label = {origin_label}'.format(origin_label=self.origin_label))

        self.origin_images, self.origin_labels = filter_by_label(self.origin_label, self.trainX, self.trainY)

        self.origin_images = np.array(self.origin_images[:self.num_images])
        self.origin_labels = np.array(self.origin_labels[:self.num_images])

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
        self.file_shared_name = self.method_name + '_' + classifier_name + f'_{origin_label}_{self.target_label}' + 'weight=' + str(
            self.weight).replace('.', ',') + '_' + str(self.num_images)

        self.autoencoder_file_name = self.file_shared_name + 'autoencoder' + '.h5'
        self.result_file_name = self.file_shared_name + 'result' + '.txt'

        logger.debug('combining target_labels for autoencoder training')

        self.combined_labels = combined_function(self.border_origin_images, self.internal_origin_images,
                                                 self.origin_images)
        self.autoencoder = None
        self.generated_borders = None
        self.generated_candidates = None
        self.adv_result = np.array([])
        self.origin_adv_result = None
        self.smooth_adv = None
        self.L0_befores = None
        self.L0_afters = None
        self.L2_befores = None
        self.L2_afters = None
        logger.debug('init attacking DONE!')

    def autoencoder_attack(self, loss):
        ae_trainee = MnistAutoEncoder()
        autoencoder_path = os.path.join(SAVED_ATTACKER_PATH, self.method_name, self.autoencoder_file_name)

        if os.path.isfile(autoencoder_path):
            logger.debug(
                'found pre-trained autoencoder for: origin_label = {origin_label}, target_label = {target_label}'.format(
                    origin_label=self.origin_label, target_label=self.target_label))

            self.autoencoder = tf.keras.models.load_model(
                autoencoder_path,
                compile=False)
            adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
            # self.autoencoder.compile(optimizer=adam,
            #                          loss=loss(self.classifier, self.target_vector, self.weight))
            self.end_time = time.time()


        else:
            logger.debug(
                'not found pre-trained autoencoder for: origin_label = {origin_label}, target_label = {target_label}'.format(
                    origin_label=self.origin_label, target_label=self.target_label))
            logger.debug('training autoencoder for: origin_label={origin_label}, target_label={target_label}'.format(
                origin_label=self.origin_label, target_label=self.target_label))

            self.autoencoder = ae_trainee.get_architecture()
            adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.autoencoder.compile(optimizer=adam,
                                     loss=loss(self.classifier, self.target_vector, self.weight))
            # self.autoencoder.compile(optimizer=adam, loss=tf.keras.losses.binary_crossentropy)
            early_stopping = EarlyStopping(monitor='loss', verbose=0, mode='min')
            model_checkpoint = ModelCheckpoint(autoencoder_path,
                                               save_best_only=True, monitor='loss',
                                               mode='min')

            history = self.autoencoder.fit(self.origin_images, self.origin_images, epochs=500, batch_size=512,
                                           callbacks=[early_stopping, model_checkpoint], verbose=1)
            self.optimal_epoch = len(history.history['loss'])
            logger.debug('training autoencoder DONE!')

        self.generated_candidates = self.autoencoder.predict(self.origin_images)
        # self.adv_result, _, self.origin_adv_result, _ = filter_candidate_adv(self.origin_images,
        #                                                                      self.generated_candidates,
        #                                                                      self.target_label,
        #                                                                      cnn_model=self.classifier)
        # # self.smooth_adv, self.L0_befores, self.L0_afters, self.L2_befores, self.L2_afters = smooth_adv_border_V3(
        #     self.classifier, self.adv_result[:-1], self.origin_adv_result[:-1],
        #     self.target_label, step=self.step)
        if self.adv_result is None:
            return
        if self.adv_result.shape[0] == 0:
            return
        # self.smooth_adv, self.L0_befores, self.L0_afters, self.L2_befores, self.L2_afters = smooth_adv_border_V3(
        #     self.classifier, self.adv_result[:-1], self.origin_adv_result[:-1], self.target_label, step=self.step)
        self.L0_afters = []
        self.L2_afters = []
        for adv, ori in zip(self.adv_result, self.origin_adv_result):
            self.L0_afters.append(compute_l0_V2(adv, ori))
            self.L2_afters.append(compute_l2_V2(adv, ori))
        self.L0_afters, self.L2_afters = np.array(self.L0_afters), np.array(self.L2_afters)
        logger.debug(f'adv shape {self.adv_result.reshape}')

    def export_result(self):
        # result = '<=========='
        result = ''
        if self.smooth_adv is not None:
            str_smooth_adv = list(map(str, self.smooth_adv))
            result += '\n' + '\n'.join(str_smooth_adv)
        if self.adv_result is None or self.adv_result.shape[0] == 0:
            return 0, [], []

        f = open(os.path.join('result', self.method_name, self.file_shared_name + 'step=' + str(self.step) + '.txt', ),
                 'w')
        f.write(result)
        f.close()
        #
        # L0_before_txt = np.array2string(self.L0_befores, separator=' ')
        # L0_before_txt = L0_before_txt.replace('[', '')
        # L0_before_txt = L0_before_txt.replace(']', '')
        # L0_before_txt = L0_before_txt.replace(' ', '\n')

        L0_after_txt = np.array2string(self.L0_afters, separator=' ')
        L0_after_txt = L0_after_txt.replace(']', '')
        L0_after_txt = L0_after_txt.replace('[', '')
        L0_after_txt = L0_after_txt.replace(' ', '\n')

        # L2_before_txt = np.array2string(self.L2_befores, separator=' ')
        # L2_before_txt = L2_before_txt.replace('[', '')
        # L2_before_txt = L2_before_txt.replace(']', '')
        # L2_before_txt = L2_before_txt.replace(' ', '\n')

        L2_after_txt = np.array2string(self.L2_afters, separator=' ')
        L2_after_txt = L2_after_txt.replace('[', '')
        L2_after_txt = L2_after_txt.replace(']', '')
        L2_after_txt = L2_after_txt.replace(' ', '\n')

        # f = open(os.path.join(RESULT_FOLDER_PATH, self.method_name,
        #                       self.file_shared_name + 'step=' + str(self.step) + 'L0_before.txt'), 'w')
        # f.write(L0_before_txt)
        # f.close()

        f = open(os.path.join(RESULT_FOLDER_PATH, self.method_name,
                              self.file_shared_name + 'step=' + str(self.step) + 'L0_after.txt'), 'w')
        f.write(L0_after_txt)
        f.close()

        # f = open(os.path.join(RESULT_FOLDER_PATH, self.method_name,
        #                       self.file_shared_name + 'step=' + str(self.step) + 'L2_before.txt'), 'w')
        # f.write(L2_before_txt)
        # f.close()

        f = open(os.path.join(RESULT_FOLDER_PATH, self.method_name,
                              self.file_shared_name + 'step=' + str(self.step) + 'L2_after.txt'), 'w')
        f.write(L2_after_txt)
        f.close()

        # return result, self.end_time - self.start_time, self.L0_afters, self.L2_afters
        return self.adv_result.shape[0] / float(self.num_images), self.L0_afters, self.L2_afters


def run_thread_V2(classifier_name, trainX, trainY):
    logger.debug("\n=======================================================")
    logger.debug('processing model: ' + classifier_name)
    cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    result_txt = classifier_name + '\n'
    # AE_LOSS = AE_LOSSES.border_loss
    weight_result = []
    L0s = []
    L2s = []
    for weight_index in range(1, 11):
        weight_value = weight_index * 0.1
        # weight_value = weight_index
        weight_result_i = []
        for origin_label in range(9, 10):
            weight_result_i_j = []
            for target_position in range(2, 3):
                attacker = AAE_V2(origin_label, np.array(trainX), np.array(trainY), cnn_model,
                                  target_position=target_position, classifier_name=classifier_name,
                                  weight=weight_value)
                attacker.autoencoder_attack(loss=AE_LOSSES.re_rank_loss)
                sucess_rate_i, L0, L2 = attacker.export_result()
                weight_result_i_j.append(sucess_rate_i)
                if len(L0) != 0:
                    for L0_i, L2_i in zip(L0, L2):
                        L0s.append(L0_i)
                        L2s.append(L2_i)
                del attacker
            weight_result_i.append(weight_result_i_j)
        weight_result_i = np.array(weight_result_i)
        np.savetxt(f'./result/aae/{classifier_name}_{weight_value}.csv', weight_result_i, delimiter=",")

        weight_result_i = np.average(weight_result_i, axis=0)
        weight_result.append(weight_result_i)

    weight_result = np.array(weight_result)
    s = np.array2string(weight_result, separator=' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    f = open('./result/aae/' + classifier_name + '.txt', 'w')
    f.write(s)
    f.close()

    L0s = np.array(L0s).flatten()
    L2s = np.array(L2s).flatten()
    if L0s.shape[0] == 0:
        return
    L0s = reject_outliers_v2(L0s)
    L2s = reject_outliers_v2(L2s)

    min_l0, max_l0, avg_l0 = np.min(L0s), np.max(L0s), np.average(L0s)
    min_l2, max_l2, avg_l2 = np.min(L2s), np.max(L2s), np.average(L2s)

    l0_l2_txt = f'L0: {min_l0}, {max_l0}, {avg_l0}\nL2: {min_l2}, {max_l2}, {avg_l2}'
    f = open('./result/aae/' + classifier_name + 'l0_l2.txt', 'w')
    f.write(l0_l2_txt)
    f.close()


def run_thread_V1(classifier_name, trainX, trainY):
    logger.debug("\n=======================================================")
    logger.debug('processing model: ' + classifier_name)
    cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    result_txt = classifier_name + '\n'
    # AE_LOSS = AE_LOSSES.border_loss
    weight_result = []
    for weight_index in [0.05, 0.1, 0.5, 1.0]:
        weight_value = weight_index
        # weight_value = weight_index
        weight_result_i = []
        for origin_label in range(0, 10):
            weight_result_i_j = []
            for target_position in range(2, 11):
                attacker = AAE_V2(origin_label, np.array(trainX), np.array(trainY), cnn_model,
                                  target_position=target_position, classifier_name=classifier_name,
                                  weight=weight_value)
                attacker.autoencoder_attack(loss=AE_LOSSES.re_rank_loss)
                sucess_rate_i, _, _ = attacker.export_result()
                weight_result_i_j.append(sucess_rate_i)
                del attacker
            weight_result_i.append(weight_result_i_j)
        weight_result_i = np.array(weight_result_i)
        np.savetxt(f'./result/aae/{classifier_name}_{weight_value}.csv', weight_result_i, delimiter=",")

        weight_result_i = np.average(weight_result_i, axis=0)
        weight_result.append(weight_result_i)

    weight_result = np.array(weight_result)
    s = np.array2string(weight_result, separator=' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    f = open('./result/aae/' + classifier_name + '.txt', 'w')
    f.write(s)
    f.close()


class MyThread(threading.Thread):
    def __init__(self, classifier_name, trainX, trainY):
        super(MyThread, self).__init__()
        self.classifier_name = classifier_name
        self.trainX = trainX
        self.trainY = trainY

    def run(self):
        run_thread_V2(self.classifier_name, self.trainX, self.trainY)


if __name__ == '__main__':
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

    logger.debug('starting multi-thread')
    thread1 = MyThread(pretrained_model_name[0], trainX, trainY)
    thread2 = MyThread(pretrained_model_name[1], trainX, trainY)
    thread3 = MyThread(pretrained_model_name[2], trainX, trainY)
    thread4 = MyThread(pretrained_model_name[3], trainX, trainY)

    thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()

    thread1.join()
    # thread2.join()
    # thread3.join()
    # thread4.join()

    logger.debug('Exiting Main Thread')
    logger.debug('robustness testing DONE !')
