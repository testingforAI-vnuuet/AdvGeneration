"""
Created At: 14/07/2021 15:39
"""
import time

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from attacker.autoencoder import MnistAutoEncoder
from attacker.constants import *
from attacker.losses import AE_LOSSES
from data_preprocessing.mnist import MnistPreprocessing
from utility.constants import *
from utility.filters.filter_advs import smooth_adv_border_V3
from utility.statistics import *
from utility.utils import *

tf.config.experimental_run_functions_eagerly(True)

logger = MyLogger.getLog()


class HPBA:
    def __init__(self, origin_label, trainX, trainY, classifier, weight, target_label=None, target_position=2,
                 classifier_name=NONAME,
                 step_to_recover=12, num_images_to_attack=1000, pattern=ALL_PATTERN, num_class=MNIST_NUM_CLASSES):
        self.origin_label = origin_label
        self.trainX = trainX
        self.trainY = trainY
        self.classifier = classifier
        self.weight = weight
        self.classifier_name = classifier_name
        self.step_to_recover = step_to_recover
        self.num_images_to_attack = num_images_to_attack

        self.target_label = target_label
        self.target_position = target_position
        self.pattern = pattern
        self.method_name = HPBA_METHOD_NAME
        self.num_images_to_train = 1000
        self.num_class = num_class

        self.origin_images, self.origin_labels = filter_by_label(label=self.origin_label, data_set=self.trainX,
                                                                 label_set=self.trainY)

        if self.target_label is None:
            self.target_label = label_ranking(self.origin_images, self.classifier)[-1 * self.target_position]

        self.target_vector = tf.keras.utils.to_categorical(self.target_label, self.num_class, dtype='float32')

        self.file_shared_name = '{method_name}_{classifier_name}_ori={origin_label}_tar={target_label}_weight={weight}_num={num_images}'.format(
            method_name=self.method_name, classifier_name=self.classifier_name, origin_label=self.origin_label,
            target_label=self.target_label, weight=str(self.weight).replace('.', ','),
            num_images=self.num_images_to_train)

        self.general_result_folder = os.path.abspath(os.path.join(RESULT_FOLDER_PATH, self.method_name))
        self.autoencoder_folder = os.path.join(self.general_result_folder, TEXT_AUTOENCODER)
        self.images_folder = os.path.join(self.general_result_folder, TEXT_IMAGE)
        self.result_summary_folder = os.path.join(self.general_result_folder, TEXT_RESULT_SUMMARY)
        self.data_folder = os.path.join(self.general_result_folder, TEXT_DATA)
        mkdirs([self.general_result_folder, self.autoencoder_folder, self.images_folder, self.result_summary_folder,
                self.data_folder])

        self.autoencoder = None
        self.autoencoder_file_path = os.path.join(self.autoencoder_folder,
                                                  self.file_shared_name + '_' + TEXT_AUTOENCODER + '.h5')

        self.adv_result = None
        self.adv_result_path = os.path.join(self.data_folder,
                                            self.file_shared_name + '_adv_' + get_timestamp() + '.npy')
        self.origin_adv_result = None
        self.origin_adv_result_path = os.path.join(self.data_folder,
                                                   self.file_shared_name + '_origin_' + get_timestamp() + '.npy')
        self.optimal_epoch = None
        self.smooth_adv_speed = None
        self.optimized_adv = None
        self.optimized_adv_path = os.path.join(self.data_folder,
                                               self.file_shared_name + '_optimized_adv_' + get_timestamp() + '.npy')
        self.summary_path = os.path.join(self.result_summary_folder,
                                         self.file_shared_name + '_summary_' + get_timestamp() + '.txt')
        self.L0_befores = None
        self.L0_afters = None
        self.L2_befores = None
        self.L2_afters = None

        self.start_time = None
        self.end_time = None

    def autoencoder_attack(self):
        ae_trainee = MnistAutoEncoder()
        self.start_time = time.time()
        if check_path_exists(self.autoencoder_file_path):
            logger.debug(
                'found pre-trained autoencoder for: origin_label = {origin_label}, target_label = {target_label}'.format(
                    origin_label=self.origin_label, target_label=self.target_label))
            self.autoencoder = tf.keras.models.load_model(self.autoencoder_file_path, compile=False)

        else:
            logger.debug(
                'not found pre-trained autoencoder for: origin_label = {origin_label}, target_label = {target_label}'.format(
                    origin_label=self.origin_label, target_label=self.target_label))
            logger.debug('training autoencoder for: origin_label={origin_label}, target_label={target_label}'.format(
                origin_label=self.origin_label, target_label=self.target_label))
            self.autoencoder = ae_trainee.get_architecture()
            adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.autoencoder.compile(optimizer=adam,
                                     loss=AE_LOSSES.cross_entropy_loss(self.classifier, self.target_vector,
                                                                       self.weight))
            early_stopping = EarlyStopping(monitor='loss', verbose=0, mode='min')
            model_checkpoint = ModelCheckpoint(self.autoencoder_file_path,
                                               save_best_only=True, monitor='loss',
                                               mode='min')

            history = self.autoencoder.fit(self.origin_images[:self.num_images_to_train],
                                           self.origin_images[:self.num_images_to_train], epochs=500, batch_size=256,
                                           callbacks=[early_stopping, model_checkpoint], verbose=1)
            self.optimal_epoch = len(history.history['loss'])

        generated_candidates = self.autoencoder.predict(self.origin_images[:self.num_images_to_attack])
        self.adv_result, _, self.origin_adv_result, _ = filter_candidate_adv(
            self.origin_images[:self.num_images_to_attack], generated_candidates, self.target_label,
            cnn_model=self.classifier)
        self.optimized_adv, self.smooth_adv_speed, self.L0_befores, self.L0_afters, self.L2_befores, self.L2_afters = smooth_adv_border_V3(
            self.classifier, self.adv_result, self.origin_adv_result,
            self.target_label, step=self.step_to_recover, return_adv=True)
        self.end_time = time.time()
        np.save(self.adv_result_path, self.adv_result)
        np.save(self.origin_adv_result_path, self.origin_adv_result)
        np.save(self.optimized_adv)

        self.L0_befores, self.L2_befores = compute_distance(self.adv_result, self.origin_adv_result)
        self.L0_afters, self.L2_afters = compute_distance(self.optimized_adv, self.origin_adv_result)

    def export_result(self):
        if self.adv_result is None:
            self.autoencoder_attack()
        logger.debug('exporting results')
        result = ''
        result += 'success_rate: ' + str(self.adv_result.shape[0] / self.num_images_to_attack)
        result += '\n'
        result += 'L0 distance (min/max/avg): ' + '{min_l0}/{max_l0}/{avg_l0}'.format(min_l0=np.min(self.L0_afters),
                                                                                      max_l0=np.max(self.L0_afters),
                                                                                      avg_l0=round(
                                                                                          np.average(self.L0_afters),
                                                                                          2))
        result += '\n'
        result += 'L2 distance (min/max/avg): ' + '{min_l2}/{max_l2}/{avg_l2}'.format(min_l2=np.min(self.L2_afters),
                                                                                      max_l2=np.max(self.L2_afters),
                                                                                      avg_l2=round(
                                                                                          np.average(self.L2_afters),
                                                                                          2))
        result += '\n'
        result += 'exe_time(second): ' + str(self.end_time - self.start_time)

        write_to_file(content=result, path=self.summary_path)


if __name__ == '__main__':
    logger.debug('pre-processing data')
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

    trainX, trainY = MnistPreprocessing.quick_preprocess_data(trainX, trainY, num_classes=MNIST_NUM_CLASSES,
                                                              rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                              chl=MNIST_IMG_CHL)
    testX, testY = MnistPreprocessing.quick_preprocess_data(testX, testY, num_classes=MNIST_NUM_CLASSES,
                                                            rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                            chl=MNIST_IMG_CHL)

    np.save('../../data/mnist/mnist_training.npy', trainX)
    np.save('../../data/mnist/mnist_label.npy', trainY)
    # logger.debug('pre-processing data DONE !')
    # classifier = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/Alexnet.h5')
    #
    # logger.debug('robustness testing start')
    # attacker = HPBA(origin_label=9, target_label=None, target_position=2, classifier=classifier, weight=0.5,
    #                 trainX=trainX, trainY=trainY, classifier_name='Alexnet')
    # attacker.autoencoder_attack()
    # attacker.export_result()
