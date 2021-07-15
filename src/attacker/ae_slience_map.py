"""
Created At: 11/06/2021 09:43
"""

import os
import threading
import time
from collections import defaultdict

from attacker.ae_custom_layer import *
from attacker.constants import *
from attacker.losses import AE_LOSSES
from attacker.mnist_utils import reject_outliers_v2
from data_preprocessing.mnist import MnistPreprocessing
from utility.constants import *
from utility.filters.filter_advs import smooth_adv_border_V3
from utility.statistics import *

tf.config.experimental_run_functions_eagerly(True)

logger = MyLogger.getLog()
pretrained_model_name = ['Alexnet', 'Lenet_v2', 'vgg13', 'vgg16']


class ae_slience_map:
    def __init__(self, trainX, trainY, origin_label, target_position, classifier, weight, saved_ranking_features_file,
                 classifier_name='noname',
                 num_images=1000, target_label=None, pre_softmax_layer_name='pre_softmax_layer', num_features=10,
                 step=6):
        """

        :param trainX:
        :type trainX:
        :param trainY:
        :type trainY:
        :param origin_label:
        :type origin_label:
        :param target_position:
        :type target_position:
        :param classifier:
        :type classifier:
        :param weight:
        :type weight:
        :param classifier_name:
        :type classifier_name:
        :param num_images:
        :type num_images:
        """
        self.trainX, self.trainY = trainX, trainY
        self.origin_label, self.target_position = origin_label, target_position
        self.classifier, self.classifier_name = classifier, classifier_name
        self.weight, self.num_images = weight, num_images
        self.method_name = SLIENCE_MAP_METHOD_NAME

        logger.debug('init attacking: origin_label = {origin_label}'.format(origin_label=self.origin_label))
        # for training autoenencoder
        self.optimal_epoch = None

        self.origin_images, self.origin_labels = filter_by_label(self.origin_label, self.trainX, self.trainY)
        self.origin_images = np.array(self.origin_images[:self.num_images])
        self.origin_labels = np.array(self.origin_labels[:self.num_images])
        logger.debug('shape of origin_images: {shape}'.format(shape=self.origin_images.shape))
        logger.debug('shape of origin_labels: {shape}'.format(shape=self.origin_labels.shape))

        if self.target_position is None:
            self.target_label = target_label
        else:
            self.target_label = label_ranking(self.origin_images, self.classifier)[-1 * self.target_position]
        self.target_vector = tf.keras.utils.to_categorical(self.target_label, MNIST_NUM_CLASSES, dtype='float32')
        logger.debug('target_label: {target_label}'.format(target_label=self.target_label))
        #
        # self.classifier_pre_softmax = tf.keras.models.Model(inputs=self.classifier.input,
        #                                                     ouputs=self.classifier.get_layer(
        #                                                         pre_softmax_layer_name).output)
        self.file_shared_name = self.method_name + '_' + classifier_name + f'_{origin_label}_{self.target_label}' + 'weight=' + str(
            self.weight).replace('.', ',') + '_' + str(self.num_images)

        self.num_features = num_features
        self.saved_ranking_features_file = saved_ranking_features_file
        # self.important_features = self.__get_important_features()
        self.important_features = [0]*100
        self.autoencoder = None
        self.autoencoder_file_path = os.path.join(SAVED_ATTACKER_PATH, self.method_name,
                                                  self.file_shared_name + 'autoencoder.h5')

        self.generated_candidates = None
        self.adv_result = None
        self.origin_adv_result_label = None
        self.start_time = None
        self.end_time = None
        self.adv_file_path = os.path.join(RESULT_FOLDER_PATH, self.method_name, self.file_shared_name + 'adv.npy')
        self.origin_file_path = os.path.join(RESULT_FOLDER_PATH, self.method_name,
                                             self.file_shared_name + 'origin.npy')
        self.smooth_adv = None
        self.step = step
        self.L0_befores = None
        self.L0_afters = None
        self.L2_befores = None
        self.L2_afters = None

    def autoencoder_attack(self, input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)):
        self.start_time = time.time()
        custom_objects = {'concate_start_to_end': concate_start_to_end}

        if os.path.isfile(self.autoencoder_file_path):
            logger.debug('[training] found pre-trained autoencoder')
            with keras.utils.custom_object_scope(custom_objects):
                self.autoencoder = tf.keras.models.load_model(self.autoencoder_file_path, compile=False)
        else:

            logger.debug(
                '[training] training autoencoder for: origin_label={origin_label}, target_label={target_label}'.format(
                    origin_label=self.origin_label, target_label=self.target_label))

            input_img = keras.layers.Input(shape=input_shape)
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
            x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
            # Dense
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
            x = keras.layers.UpSampling2D((2, 2))(x)
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
            decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

            decoded = concate_start_to_end(important_pixels=self.important_features)([decoded, input_img])
            self.autoencoder = keras.models.Model(input_img, decoded)
            adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.autoencoder.compile(optimizer=adam,
                                     loss=AE_LOSSES.cross_entropy_loss(self.classifier, self.target_vector,
                                                                       self.weight),
                                     run_eagerly=True)

            # early_stopping = EarlyStopping(monitor='loss', verbose=0, mode='min')
            # model_checkpoint = ModelCheckpoint(self.autoencoder_file_path,
            #                                    save_best_only=True, monitor='loss',
            #                                    mode='min')
            # history = self.autoencoder.fit(self.origin_images, self.origin_images, epochs=500, batch_size=512,
            #                                callbacks=[early_stopping, model_checkpoint], verbose=1)
            history = self.autoencoder.fit(self.origin_images, self.origin_images, epochs=50, batch_size=256,
                                           verbose=1)
            self.autoencoder.save(self.autoencoder_file_path)
            self.optimal_epoch = len(history.history['loss'])

        self.generated_candidates = self.autoencoder.predict(self.origin_images)
        self.adv_result, _, self.origin_adv_result, self.origin_adv_result_label = filter_candidate_adv(
            self.origin_images,
            self.generated_candidates,
            self.target_label,
            cnn_model=self.classifier)

        self.end_time = time.time()

        self.smooth_adv, self.L0_befores, self.L0_afters, self.L2_befores, self.L2_afters = smooth_adv_border_V3(
            self.classifier, self.adv_result[:-1], self.origin_adv_result[:-1],
            self.target_label, step=self.step)

        logger.debug('[training] sucess_rate={sucess_rate}'.format(sucess_rate=self.adv_result.shape))
        # np.save(self.adv_file_path, self.adv_result)
        # np.save(self.origin_file_path, self.origin_adv_result)
        logger.debug('[training] training autoencoder DONE!')

    def __get_important_features(self):
        ranking_matrix = None
        if not os.path.isfile(self.saved_ranking_features_file):
            logger.debug(self.saved_ranking_features_file)
            logger.error('not found ranking matrix')
            raise NotImplementedError('not found ranking matrix')
        if os.path.isfile(self.saved_ranking_features_file):
            ranking_matrix = np.load(self.saved_ranking_features_file)
        ranking_matrix = ranking_matrix.flatten()
        return np.argsort(ranking_matrix)[::-1][:self.num_features]

    def export_result(self):
        # result = '<=========='
        result = ''
        if self.smooth_adv is not None:
            str_smooth_adv = list(map(str, self.smooth_adv))
            result += '\n'.join(str_smooth_adv)
        if self.adv_result is None or self.adv_result.shape[0] == 0:
            return 0, [], []

        # f = open(os.path.join('result', self.method_name, self.file_shared_name + 'step=' + str(self.step) + '.txt', ),
        #          'w')
        # f.write(result)
        # f.close()
        #
        # L0_before_txt = np.array2string(self.L0_befores, separator=' ')
        # L0_before_txt = L0_before_txt.replace('[', '')
        # L0_before_txt = L0_before_txt.replace(']', '')
        # L0_before_txt = L0_before_txt.replace(' ', '\n')

        # L0_after_txt = np.array2string(self.L0_afters, separator=' ')
        # L0_after_txt = L0_after_txt.replace(']', '')
        # L0_after_txt = L0_after_txt.replace('[', '')
        # L0_after_txt = L0_after_txt.replace(' ', '\n')

        # L2_before_txt = np.array2string(self.L2_befores, separator=' ')
        # L2_before_txt = L2_before_txt.replace('[', '')
        # L2_before_txt = L2_before_txt.replace(']', '')
        # L2_before_txt = L2_before_txt.replace(' ', '\n')

        # L2_after_txt = np.array2string(self.L2_afters, separator=' ')
        # L2_after_txt = L2_after_txt.replace('[', '')
        # L2_after_txt = L2_after_txt.replace(']', '')
        # L2_after_txt = L2_after_txt.replace(' ', '\n')

        # f = open(os.path.join(RESULT_FOLDER_PATH, self.method_name,
        #                       self.file_shared_name + 'step=' + str(self.step) + 'L0_before.txt'), 'w')
        # f.write(L0_before_txt)
        # f.close()

        # f = open(os.path.join(RESULT_FOLDER_PATH, self.method_name,
        #                       self.file_shared_name + 'step=' + str(self.step) + 'L0_after.txt'), 'w')
        # f.write(L0_after_txt)
        # f.close()

        # f = open(os.path.join(RESULT_FOLDER_PATH, self.method_name,
        #                       self.file_shared_name + 'step=' + str(self.step) + 'L2_before.txt'), 'w')
        # f.write(L2_before_txt)
        # f.close()

        # f = open(os.path.join(RESULT_FOLDER_PATH, self.method_name,
        #                       self.file_shared_name + 'step=' + str(self.step) + 'L2_after.txt'), 'w')
        # f.write(L2_after_txt)
        # f.close()

        # return result, self.end_time - self.start_time, self.L0_afters, self.L2_afters
        return self.adv_result.shape[0] / float(self.num_images), self.L0_afters, self.L2_afters, self.smooth_adv

    #
    # def export_resultV2(self):
    #
    #     f = open(os.path.join(RESULT_FOLDER_PATH, self.method_name, self.file_shared_name + 'sucess_rate.txt'), 'w')
    #     f.write(str(self.adv_result.shape[0] / float(self.num_images)))
    #     f.close()
    #     return self.adv_result.shape[0] / float(self.num_images)


def run_thread(classifier_name, trainX, trainY):
    logger.debug("\n=======================================================")
    logger.debug('processing model: ' + classifier_name)
    cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    result_txt = classifier_name + '\n'
    # AE_LOSS = AE_LOSSES.border_loss
    weight_result = []
    L0s = []
    L2s = []
    sucess_rate_dict = defaultdict(int)
    for weight_index in [0.01, 0.05, 0.5, 1.0]:
        weight_value = weight_index
        # weight_value = weight_index
        weight_result_i = []
        for origin_label in range(0, 10):
            weight_result_i_j = []
            saved_ranking_features_file = os.path.join(RESULT_FOLDER_PATH,
                                                       f'slience_map/slience_matrix_{classifier_name}_label={origin_label},optimizer=adam,lr=0.1,lamda=0.1.npy')

            if origin_label == 9 and classifier_name == 'Lenet_v2':
                saved_ranking_features_file = os.path.join(RESULT_FOLDER_PATH,
                                                           'slience_map/slience_matrix_Lenet_v2_label=9,optimizer=adam,lr=0.5,lamda=0.1.npy')

            for target_position in range(2, 11):
                attacker = ae_slience_map(trainX=trainX, trainY=trainY, origin_label=origin_label,
                                          target_position=target_position, classifier=cnn_model, weight=weight_value,
                                          saved_ranking_features_file=saved_ranking_features_file,
                                          classifier_name=classifier_name, num_features=100)
                attacker.autoencoder_attack()
                weight_result_i_j.append(attacker.export_result())
                success_rate = attacker.export_result()
                # _, _, L0, L2 = attacker.export_resultV2()
                # L0s.append(L0)
                # L2s.append(L2)
                success_rate_single = {f'{origin_label}_{target_position}_{weight_value}': success_rate}
                sucess_rate_dict.update(success_rate_single)
                del attacker
            weight_result_i.append(weight_result_i_j)
        weight_result_i = np.array(weight_result_i)
        np.savetxt(f'./result/ae_slience_map/{classifier_name}_{weight_value}.csv', weight_result_i, delimiter=",")
        weight_result_i = np.average(weight_result_i, axis=0)
        weight_result.append(weight_result_i)

    key_max = max(sucess_rate_dict, key=sucess_rate_dict.get)
    value = sucess_rate_dict[key_max]
    result_max = f'{key_max}: {value}'

    key_min = min(sucess_rate_dict, key=sucess_rate_dict.get)
    value = sucess_rate_dict[key_min]
    result_min = f'{key_min}: {value}'
    result = result_max + '\n' + result_min

    f = open('./result/ae_slience_map/' + classifier_name + 'success_rate.txt', 'w')
    f.write(result)
    f.close()


def run_thread_V2(classifier_name, trainX, trainY):
    logger.debug("\n=======================================================")
    logger.debug('processing model: ' + classifier_name)
    cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    result_txt = classifier_name + '\n'
    # AE_LOSS = AE_LOSSES.border_loss
    weight_result = []
    L0s = []
    L2s = []
    smooth_adv_speed = []
    step = 0.3
    for weight_index in range(1, 11):
        weight_value = weight_index * 0.1
        # weight_value = weight_index
        weight_result_i = []
        for origin_label in range(9, 10):
            saved_ranking_features_file = os.path.join(RESULT_FOLDER_PATH,
                                                       f'slience_map/slience_matrix_{classifier_name}_label={origin_label},optimizer=adam,lr=0.1,lamda=0.1.npy')

            if origin_label == 9 and classifier_name == 'Lenet_v2':
                saved_ranking_features_file = os.path.join(RESULT_FOLDER_PATH,
                                                           'slience_map/slience_matrix_Lenet_v2_label=9,optimizer=adam,lr=0.5,lamda=0.1.npy')

            weight_result_i_j = []
            for target_position in range(2, 3):
                attacker = ae_slience_map(trainX=trainX, trainY=trainY, origin_label=origin_label,
                                          target_position=target_position, classifier=cnn_model, weight=weight_value,
                                          saved_ranking_features_file=saved_ranking_features_file,
                                          classifier_name=classifier_name, num_features=100)
                attacker.autoencoder_attack()
                sucess_rate_i, L0, L2, smooth_adv_i = attacker.export_result()
                weight_result_i_j.append(sucess_rate_i)

                if len(L0) != 0:
                    for L0_i, L2_i in zip(L0, L2):
                        L0s.append(L0_i)
                        L2s.append(L2_i)
                        smooth_adv_speed.append(smooth_adv_i)
                del attacker
            weight_result_i.append(weight_result_i_j)
        weight_result_i = np.array(weight_result_i)
        ranking_type = 'jsma_ka'
        np.savetxt(f'./result/slience_map/{classifier_name}_{weight_value}{ranking_type}.csv', weight_result_i,
                   delimiter=",")

        weight_result_i = np.average(weight_result_i, axis=0)
        weight_result.append(weight_result_i)

    weight_result = np.array(weight_result)
    s = np.array2string(weight_result, separator=' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    f = open('./result/slience_map/' + classifier_name + 'success_rate.txt', 'w')
    f.write(s)
    f.close()

    smooth_adv_speed = np.asarray(smooth_adv_speed)
    smooth_adv_speed = np.average(smooth_adv_speed, axis=0)
    np.savetxt(f'./result/slience_map/{classifier_name}_avg_recover_speed_step={step}{ranking_type}.csv',
               smooth_adv_speed, delimiter=',')

    L0s = np.array(L0s).flatten()
    L2s = np.array(L2s).flatten()

    L0s = reject_outliers_v2(L0s)
    L2s = reject_outliers_v2(L2s)
    if L0s.shape[0] == 0:
        return
    min_l0, max_l0, avg_l0 = np.min(L0s), np.max(L0s), np.average(L0s)
    min_l2, max_l2, avg_l2 = np.min(L2s), np.max(L2s), np.average(L2s)

    l0_l2_txt = f'L0: {min_l0}, {max_l0}, {avg_l0}\nL2: {min_l2}, {max_l2}, {avg_l2}'
    f = open('./result/slience_map/' + classifier_name + f'l0_l2_step={step}{ranking_type}.txt', 'w')
    f.write(l0_l2_txt)
    f.close()
    logger.debug('ok')


class MyThread(threading.Thread):
    def __init__(self, classifier_name, trainX, trainY):
        super(MyThread, self).__init__()
        self.classifier_name = classifier_name
        self.trainX = trainX
        self.trainY = trainY

    def run(self):
        run_thread_V2(self.classifier_name, self.trainX, self.trainY)


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

    logger.debug('starting multi-thread')
    # saved_ranking_features_file = os.path.join(RESULT_FOLDER_PATH,
    #                                            'slience_map/slience_matrix_Lenet_v2_label=9,optimizer=adam,lr=0.5,lamda=0.1.npy')
    # saved_ranking_features_file = os.path.join(RESULT_FOLDER_PATH,
    #                                            'slience_map/slience_matrix_Alexnet_label=9,optimizer=adam,lr=0.1,lamda=0.1.npy')
    thread1 = MyThread(pretrained_model_name[0], trainX, trainY)
    thread2 = MyThread(pretrained_model_name[1], trainX, trainY)
    # thread3 = MyThread(pretrained_model_name[2], trainX, trainY)
    # thread4 = MyThread(pretrained_model_name[3], trainX, trainY)

    thread1.start()
    thread2.start()
    # thread3.start()
    # thread4.start()

    thread1.join()
    thread2.join()
    # thread3.join()
    # thread4.join()

    logger.debug('Exiting Main Thread')
    logger.debug('robustness testing DONE !')
# if __name__ == '__main__':
#     (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()
#
#     trainX, trainY = MnistPreprocessing.quick_preprocess_data(trainX, trainY, num_classes=MNIST_NUM_CLASSES,
#                                                               rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
#                                                               chl=MNIST_IMG_CHL)
#     testX, testY = MnistPreprocessing.quick_preprocess_data(testX, testY, num_classes=MNIST_NUM_CLASSES,
#                                                             rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
#                                                             chl=MNIST_IMG_CHL)
#
#     classifier_name = pretrained_model_name[1]
#     classifier = tf.keras.models.load_model(os.path.join(PRETRAIN_CLASSIFIER_PATH, classifier_name + '.h5'))
#
#     saved_ranking_features_file = os.path.join(RESULT_FOLDER_PATH,
#                                                'slience_map/slience_matrix_Lenet_v2_label=9,optimizer=adam,lr=0.1,lamda=0.5.npy')
#
#     print(saved_ranking_features_file)
#     attacker = ae_slience_map(trainX=trainX, trainY=trainY, origin_label=9, target_position=2, classifier=classifier,
#                               weight=0.1, saved_ranking_features_file=saved_ranking_features_file,
#                               classifier_name=classifier_name, target_label=None, num_images=1000, num_features=100)
#     attacker.autoencoder_attack()
#
#     a = np.load(os.path.join(RESULT_FOLDER_PATH, 'ae_slience_map', 'ae_slience_map_Lenet_v2_9_4weight=0,1_1000adv.npy'))
#     print(a.shape)
#     n = 10
#     matplotlib.use('TkAgg')
#     plt.figure(figsize=(16, 4))
#     for i in range(n):
#         ax = plt.subplot(1, n, i + 1)
#         plt.imshow(a[i].reshape((28, 28)), cmap='gray')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
