"""
Created At: 11/06/2021 09:43
"""

import os
import threading
import time

from attacker.ae_custom_layer import *
from attacker.constants import SAVED_ATTACKER_PATH, PRETRAIN_CLASSIFIER_PATH, RESULT_FOLDER_PATH
from attacker.losses import AE_LOSSES
from data_preprocessing.mnist import MnistPreprocessing
from utility.constants import *
from utility.statistics import *

tf.config.experimental_run_functions_eagerly(True)

logger = MyLogger.getLog()
pretrained_model_name = ['Alexnet', 'Lenet', 'vgg13', 'vgg16']


class ae_slience_map:
    def __init__(self, trainX, trainY, origin_label, target_position, classifier, weight, saved_ranking_features_file,
                 classifier_name='noname',
                 num_images=1000, target_label=None, pre_softmax_layer_name='pre_softmax_layer', num_features=10):
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
        self.method_name = 'ae_slience_map'

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
        self.important_features = self.__get_important_features()
        self.autoencoder = None
        self.autoencoder_file_path = os.path.join(SAVED_ATTACKER_PATH, self.method_name,
                                                  self.file_shared_name + 'autoencoder.h5')

        self.generated_candidates = None
        self.adv_result = None
        self.origin_adv_result_label = None
        self.start = None
        self.end = None
        self.adv_file_path = os.path.join(RESULT_FOLDER_PATH, self.method_name, self.file_shared_name + 'adv.npy')
        self.origin_file_path = os.path.join(RESULT_FOLDER_PATH, self.method_name,
                                             self.file_shared_name + 'origin.npy')

    def autoencoder_attack(self, input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)):
        self.start = time.time()
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
                                 loss=AE_LOSSES.cross_entropy_loss(self.classifier, self.target_vector, self.weight),
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
        logger.debug('[training] sucess_rate={sucess_rate}'.format(sucess_rate=self.adv_result.shape))
        np.save(self.adv_file_path, self.adv_result)
        np.save(self.origin_file_path, self.origin_adv_result)
        logger.debug('[training] training autoencoder DONE!')

    def __get_important_features(self):
        ranking_matrix = None
        if not os.path.isfile(self.saved_ranking_features_file):
            logger.error('not found ranking matrix')
            raise NotImplementedError('not found ranking matrix')
        if os.path.isfile(self.saved_ranking_features_file):
            ranking_matrix = np.load(self.saved_ranking_features_file)
        ranking_matrix = ranking_matrix.flatten()
        return np.argsort(ranking_matrix)[::-1][:self.num_features]

    def export_result(self):

        return self.adv_result.shape[0] / float(self.num_images)


def run_thread_V2(classifier_name, trainX, trainY):
    logger.debug("\n=======================================================")
    logger.debug('processing model: ' + classifier_name)
    cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    result_txt = classifier_name + '\n'
    # AE_LOSS = AE_LOSSES.border_loss
    weight_result = []
    for weight_index in range(0, 11):
        weight_value = weight_index * 0.1
        # weight_value = weight_index
        weight_result_i = []
        for origin_label in range(9, 10):
            weight_result_i_j = []
            for target_position in range(2, 3):
                attacker = ae_slience_map(trainX=trainX, trainY=trainY, origin_label=origin_label,
                                          target_position=target_position, classifier=cnn_model, weight=weight_value,
                                          saved_ranking_features_file=saved_ranking_features_file,
                                          classifier_name=classifier_name, num_features=100)
                attacker.autoencoder_attack()
                weight_result_i_j.append(attacker.export_result())
                del attacker
            weight_result_i.append(weight_result_i_j)
        weight_result_i = np.average(weight_result_i, axis=0)
        weight_result.append(weight_result_i)

    weight_result = np.array(weight_result)
    s = np.array2string(weight_result, separator=' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    f = open('./result/ae_slience_map/' + classifier_name + '.txt', 'w')
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
    saved_ranking_features_file = os.path.join(RESULT_FOLDER_PATH,
                                               'slience_map/slience_matrix_Alexnet_label=9,optimizer=adam,lr=0.5,lamda=0.1.npy')
    thread1 = MyThread(pretrained_model_name[0], trainX, trainY)
    # thread2 = MyThread(pretrained_model_name[1], trainX, trainY)
    # thread3 = MyThread(pretrained_model_name[2], trainX, trainY)
    # thread4 = MyThread(pretrained_model_name[3], trainX, trainY)

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
#     classifier_name = pretrained_model_name[0]
#     classifier = tf.keras.models.load_model(os.path.join(PRETRAIN_CLASSIFIER_PATH, classifier_name + '.h5'))
#
#     saved_ranking_features_file = os.path.join(RESULT_FOLDER_PATH,
#                                                'slience_map/slience_matrix_Alexnet_label=9,optimizer=adam,lr=0.5,lamda=0.1.npy')
#
#     print(saved_ranking_features_file)
#     attacker = ae_slience_map(trainX=trainX, trainY=trainY, origin_label=9, target_position=2, classifier=classifier,
#                               weight=0.1, saved_ranking_features_file=saved_ranking_features_file,
#                               classifier_name=classifier_name, target_label=None, num_images=1000, num_features=100)
#     attacker.autoencoder_attack()
#
#     a = np.load(os.path.join(RESULT_FOLDER_PATH, 'ae_slience_map', 'ae_slience_map_Alexnet_9_7weight=0,1_1000adv.npy'))
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
