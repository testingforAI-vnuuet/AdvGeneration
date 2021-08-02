"""
Created At: 11/06/2021 09:43
"""

import os
import threading
import time

from attacker.ae_custom_layer import *
from attacker.constants import *
from attacker.losses import AE_LOSSES
from attacker.mnist_utils import reject_outliers_v2, compute_distance
from data_preprocessing.mnist import MnistPreprocessing
from utility.constants import *
from utility.filters.filter_advs import smooth_adv_border_V3
from utility.optimize_advs import optimize_advs
from utility.statistics import *

tf.config.experimental_run_functions_eagerly(True)

logger = MyLogger.getLog()
pretrained_model_name = ['Alexnet', 'Lenet_v2', 'vgg13', 'vgg16']


class ae_slience_map:
    def __init__(self, trainX, trainY, origin_label, target_position, classifier, weight, saved_ranking_features_file,
                 classifier_name='noname',
                 num_images=1000, target_label=None, pre_softmax_layer_name='pre_softmax_layer', num_features=10,
                 step=6, is_train=True):
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
        self.num_images_for_prediction = 4000
        if is_train is False:
            self.origin_images = np.array(self.origin_images[:self.num_images_for_prediction])
            self.origin_labels = np.array(self.origin_labels[:self.num_images_for_prediction])
        else:
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
        self.important_features = [0] * 100
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
        self.optimized_adv = None

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

        self.optimized_adv = optimize_advs(classifier=self.classifier,
                                           generated_advs=self.adv_result[:4000],
                                           origin_images=self.origin_adv_result[:4000],
                                           target_label=self.target_label,
                                           step=self.step, num_class=10)
        self.L0_afters, self.L2_afters = compute_distance(self.optimized_adv, self.origin_adv_result)
        self.L0_befores, self.L2_befores = compute_distance(self.adv_result, self.origin_adv_result)

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
            return 0, [], [], [], [], []

        return self.adv_result.shape[0] / float(self.num_images), self.L0_afters, self.L2_afters, self.smooth_adv, self.L0_befores, self.L2_befores


def run_thread_V2(classifier_name, trainX, trainY):
    logger.debug("\n=======================================================")
    logger.debug('processing model: ' + classifier_name)
    cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    result_txt = classifier_name + '\n'
    # AE_LOSS = AE_LOSSES.border_loss
    weight_result = []
    L0_afters = []
    L2_afters = []
    L0_befores = []
    L2_befores = []
    smooth_adv_speed = []
    step = 6
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

            for target_position in range(2, 3):
                attacker = ae_slience_map(trainX=trainX, trainY=trainY, origin_label=origin_label,
                                          target_position=target_position, classifier=cnn_model, weight=weight_value,
                                          saved_ranking_features_file=saved_ranking_features_file,
                                          classifier_name=classifier_name, num_features=100)
                attacker.autoencoder_attack()
                sucess_rate_i, L0_after, L2_after, smooth_adv_i, L0_before, L2_before = attacker.export_result()
                weight_result.append(sucess_rate_i)
                if len(L0_after) != 0:
                    for index in range(len(L0_after)):
                        L0_afters.append(L0_after[index])
                        L2_afters.append(L2_after[index])
                        L0_befores.append(L0_before[index])
                        L2_befores.append(L2_before[index])
                    smooth_adv_speed.append(smooth_adv_i)
                del attacker
            # weight_result_i.append(weight_result_i_j)

    # smooth_adv_speed = np.asarray(smooth_adv_speed)
    # smooth_adv_speed = np.average(smooth_adv_speed, axis=0)
    ranking_type = 'jsma'
    # np.savetxt(f'./result/slience_map/{classifier_name}_avg_recover_speed_step={step}{ranking_type}.csv',
    #            smooth_adv_speed, delimiter=',')

    L0_afters = np.array(L0_afters).flatten()
    L2_afters = np.array(L2_afters).flatten()

    L0_afters = reject_outliers_v2(L0_afters)
    L2_afters = reject_outliers_v2(L2_afters)

    L0_befores = np.array(L0_befores).flatten()
    L2_befores = np.array(L2_befores).flatten()

    L0_befores = reject_outliers_v2(L0_befores)
    L2_befores = reject_outliers_v2(L2_befores)


    if L0_afters.shape[0] == 0:
        return
    min_l0, max_l0, avg_l0 = np.min(L0_afters), np.max(L0_afters), np.average(L0_afters)
    min_l2, max_l2, avg_l2 = np.min(L2_afters), np.max(L2_afters), np.average(L2_afters)

    l0_l2_txt = f'L0: {min_l0}, {max_l0}, {avg_l0}\nL2: {min_l2}, {max_l2}, {avg_l2}'

    min_l0, max_l0, avg_l0 = np.min(L0_befores), np.max(L0_befores), np.average(L0_befores)
    min_l2, max_l2, avg_l2 = np.min(L2_befores), np.max(L2_befores), np.average(L2_befores)
    l0_l2_txt += '\n before: '
    l0_l2_txt += '\n ' + f'L0: {min_l0}, {max_l0}, {avg_l0}\nL2: {min_l2}, {max_l2}, {avg_l2}'
    l0_l2_txt += '\n' + str(weight_result)
    f = open('./result/ae_slience_map/' + classifier_name + f'l0_l2_step={step}{ranking_type}.txt', 'w')
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
