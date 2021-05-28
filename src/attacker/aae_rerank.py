from __future__ import absolute_import

import os
import time
from collections import defaultdict

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from attacker.autoencoder import MnistAutoEncoder
from attacker.losses import *
from attacker.mnist_utils import reject_outliers, L0, L2
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *

logger = MyLogger.getLog()

tf.config.experimental_run_functions_eagerly(True)

pretrained_model_name = ['Alexnet', 'Lenet', 'vgg13', 'vgg16']


class AAE:

    def __init__(self, trainX, trainY, origin_label, target_position, classifier, weight, classifier_name='noname',
                 num_images=1000):
        """

        :param target: target class in attack
        """
        self.method_name = 'aae'
        self.start_time = time.time()
        self.origin_label = origin_label
        self.classifier = classifier
        self.trainX = trainX
        self.trainY = trainY
        self.target_position = target_position
        self.weight = weight
        self.num_images = num_images

        # self.origin_images, self.origin_labels = filter_by_label(self.origin_label, self.trainX, self.trainY)

        self.origin_images = np.array(self.trainX[:self.num_images])
        self.origin_labels = np.array(self.trainY[:self.num_images])

        if self.origin_label is None:
            self.origin_label = 'all'
            self.target_label = DEFAULT_TARGET
        else:
            self.target_label = label_ranking(self.origin_images, self.classifier)[-1 * self.target_position]
        self.target_vector = tf.keras.utils.to_categorical(self.target_label, MNIST_NUM_CLASSES, dtype='float32')

        self.autoencoder = None
        self.file_shared_name = self.method_name + '_' + classifier_name + f'_{self.origin_label}_{self.target_label}' + 'weight=' + str(
            self.weight).replace('.', ',') + '_' + str(self.num_images)

        self.autoencoder_file_name = self.file_shared_name + 'autoencoder' + '.h5'
        self.optimal_epoch = 0
        self.generated_candidates = None
        self.adv_result = None
        self.origin_adv_result = None
        self.end_time = None
        self.adv_result_file_path = self.file_shared_name + '_adv_result' + '.npy'
        self.origin_adv_result_file_path = self.file_shared_name + '_origin_adv_result' + '.npy'
        self.num_avg_redundant_pixels = 0
        self.smooth_adv = None

        self.transfer_rate_results = defaultdict()
        self.hidden_models_name = None
        # self.target = target

    # self.target_label_onehot = keras.utils.to_categorical(target, nClasses, dtype='float32')
    # self.train_data = train_data
    # self.is_compiled = False

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
            self.autoencoder.compile(optimizer=adam,
                                     loss=loss(self.classifier, self.target_vector, self.weight))
            self.end_time = time.time()
        else:
            self.autoencoder = ae_trainee.get_architecture()
            adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.autoencoder.compile(optimizer=adam,
                                     loss=loss(self.classifier, self.target_vector, self.weight))
            # self.autoencoder.compile(optimizer=adam, loss=tf.keras.losses.binary_crossentropy)
            early_stopping = EarlyStopping(monitor='loss', verbose=0, mode='min')
            model_checkpoint = ModelCheckpoint(
                autoencoder_path,
                save_best_only=True, monitor='loss',
                mode='min')

            history = self.autoencoder.fit(self.origin_images, self.origin_images, epochs=500, batch_size=512,
                                           callbacks=[early_stopping, model_checkpoint], verbose=1)
            self.optimal_epoch = len(history.history['loss'])

        self.generated_candidates = self.autoencoder.predict(self.origin_images)
        self.adv_result, _, self.origin_adv_result, _ = filter_candidate_adv(self.origin_images,
                                                                             self.generated_candidates,
                                                                             self.target_label,
                                                                             cnn_model=self.classifier)

        # np.save(os.path.join(SAVED_NPY_PATH, self.method_name, self.adv_result_file_path), self.adv_result)
        # np.save(os.path.join(SAVED_NPY_PATH, self.method_name, self.origin_adv_result_file_path),
        #         self.origin_adv_result)
        # self.smooth_adv = smooth_adv_border_V2(self.classifier, self.adv_result[:5], self.origin_adv_result[:5], self.target_label)
        self.end_time = time.time()

    def black_box_attack(self, hidden_models_name):

        transfer_rate_results = defaultdict()
        self.hidden_models_name = hidden_models_name
        for hidden_model_name in hidden_models_name:
            hidden_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + hidden_model_name + '.h5')
            advs, _, _, _ = filter_candidate_adv(self.origin_adv_result,
                                                 self.adv_result,
                                                 self.target_label,
                                                 cnn_model=hidden_model)
            transfer_rate_results[hidden_model_name] = advs.shape[0] / float(self.adv_result.shape[0])
        self.transfer_rate_results = transfer_rate_results

    def export_result(self):
        result = '<=========='
        # str_smooth_adv = list(map(str, self.smooth_adv))
        # result += '\n' + '\n'.join(str_smooth_adv)
        result += '\norigin=' + str(self.origin_label) + ',target=' + str(self.target_label) + '\n'
        result += '\n\tweight=' + str(self.weight)
        result += '\n\t#adv=' + str(self.adv_result.shape[0])
        result += '\n\t#optimal_epoch=' + str(self.optimal_epoch)
        # result += '\n\t#avg_redundant_pixels=' + str(self.num_avg_redundant_pixels)
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

        transfer_txt = ''
        if self.hidden_models_name is not None:
            transfer_txt = '\ntransferable rate: '
            for item in self.transfer_rate_results:
                transfer_txt += f'{item} ({self.transfer_rate_results[item]}) '
        result += transfer_txt
        result += '\n==========>\n'

        f = open(os.path.join('result', self.method_name, self.file_shared_name + '.txt', ), 'w')
        f.write(result)
        f.close()

        return result, self.end_time - self.start_time

    def save_images(self):

        l2 = np.array([L2(gen, test) for gen, test in zip(self.adv_result, self.origin_adv_result)])
        # l2 = reject_outliers(l2)
        l0 = np.array([L0(gen, test) for gen, test in zip(self.adv_result, self.origin_adv_result)])
        # l0 = reject_outliers(l0)
        if l2.shape[0] == 0:
            return

        origin_adv_result_borders = get_border(self.origin_adv_result)

        sum_adv_borders = [np.sum(origin_adv_result_border.flatten()) for origin_adv_result_border in
                           origin_adv_result_borders]

        l2_argsort = np.argsort(l2)
        worst_l2_index = l2_argsort[-1]
        best_l2_index = l2_argsort[0]

        l0_avg = l0 / sum_adv_borders
        l0_argsort = np.argsort(l0_avg)
        worst_l0_index = l0_argsort[-1]
        best_l0_index = l0_argsort[0]

        l2_image_file_name = self.file_shared_name + '_l2' + '.png'
        l0_image_file_name = self.file_shared_name + '_l0' + '.png'

        path_l2 = os.path.join(SAVED_IMAGE_SAMPLE_PATH, self.method_name, l2_image_file_name)
        path_l0 = os.path.join(SAVED_IMAGE_SAMPLE_PATH, self.method_name, l0_image_file_name)

        # show for l2
        origin_image_worst_l2 = self.origin_adv_result[worst_l2_index]
        origin_image_best_l2 = self.origin_adv_result[best_l2_index]

        gen_image_worst_l2 = self.adv_result[worst_l2_index]
        gen_image_best_l2 = self.adv_result[best_l2_index]

        l2_worst = l2[worst_l2_index]
        l2_best = l2[best_l2_index]

        l0_l2_worst = l0[worst_l2_index]
        l0_l2_best = l0[best_l2_index]

        plot_images(origin_image_worst_l2, origin_image_best_l2, gen_image_worst_l2, gen_image_best_l2, l2_worst,
                    l2_best, l0_l2_worst, l0_l2_best, path_l2, self.classifier, worst_l2_index, worst_l0_index)

        # show for l0
        origin_image_worst_l0 = self.origin_adv_result[worst_l0_index]
        origin_image_best_l0 = self.origin_adv_result[best_l0_index]

        gen_image_worst_l0 = self.adv_result[worst_l0_index]
        gen_image_best_l0 = self.adv_result[best_l0_index]

        l0_worst = l0[worst_l0_index]
        l0_best = l0[best_l0_index]

        l2_l0_worst = l2[worst_l0_index]
        l2_l0_best = l2[best_l0_index]

        plot_images(origin_image_worst_l0, origin_image_best_l0, gen_image_worst_l0, gen_image_best_l0, l2_l0_worst,
                    l2_l0_best, l0_worst, l0_best, path_l0, self.classifier, worst_l0_index, best_l0_index)


if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 1000
    DEFAULT_TARGET = 7
    DEFAULT_EPSILON = 0.005
    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    AE_MODEL = CLASSIFIER_PATH + '/autoencoder_rerank_mnist.h5'

    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

    trainX, trainY = MnistPreprocessing.quick_preprocess_data(trainX, trainY, num_classes=MNIST_NUM_CLASSES,
                                                              rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                              chl=MNIST_IMG_CHL)
    testX, testY = MnistPreprocessing.quick_preprocess_data(testX, testY, num_classes=MNIST_NUM_CLASSES,
                                                            rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                            chl=MNIST_IMG_CHL)

    classifier = keras.models.load_model(ATTACKED_CNN_MODEL)

    ae4dnn_attack = AAE(trainX=trainX, trainY=trainY, origin_label=None, target_position=None, classifier=classifier,
                           weight=DEFAULT_EPSILON, classifier_name='targetmodel', num_images=1000)

    ae4dnn_attack.autoencoder_attack(loss=AE_LOSSES.re_rank_loss)
    ae4dnn_attack.black_box_attack(pretrained_model_name)
    ae4dnn_attack.export_result()
