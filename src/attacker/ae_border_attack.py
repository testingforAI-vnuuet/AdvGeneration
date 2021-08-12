"""
Created At: 23/03/2021 16:22
"""
import os.path
import threading
import time

from attacker.autoencoder import *
from attacker.constants import *
from attacker.mnist_utils import *
from utility.optimize_advs import optimize_advs
from utility.statistics import *

tf.config.experimental_run_functions_eagerly(True)

logger = MyLogger.getLog()

pretrained_model_name = ['Alexnet', 'Lenet_v2', 'vgg13', 'vgg16']


def combined_function(set1, set2, set3):
    return np.array([list(combined) for combined in zip(set1, set2, set3)])


class AutoencoderBorder:
    def __init__(self, origin_label, trainX, trainY, classifier, weight, target_position=2, classifier_name='noname',
                 step=6,
                 num_images=1000, is_train=True):
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
        self.method_name = BORDER_METHOD_NAME
        self.step = step

        logger.debug('init attacking: origin_label = {origin_label}'.format(origin_label=self.origin_label))

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
        self.file_shared_name = self.method_name + '_simpleae_' + classifier_name + f'_{origin_label}_{self.target_label}' + 'weight=' + str(
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
        self.optimized_adv = None
        logger.debug('init attacking DONE!')

    def autoencoder_attack(self, loss):
        ae_trainee = MnistAutoEncoder()
        autoencoder_path = os.path.join(SAVED_ATTACKER_PATH, self.method_name, self.autoencoder_file_name)
        self.start_time = time.time()
        if os.path.isfile(autoencoder_path) and False:
            logger.debug(
                'found pre-trained autoencoder for: origin_label = {origin_label}, target_label = {target_label}'.format(
                    origin_label=self.origin_label, target_label=self.target_label))

            self.autoencoder = tf.keras.models.load_model(
                autoencoder_path,
                compile=False)
            adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.autoencoder.compile(optimizer=adam,
                                     loss=loss(self.classifier, self.target_vector, self.origin_images, self.weight))
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
            #
            # history = self.autoencoder.fit(self.origin_images, self.combined_labels, epochs=500, batch_size=512,
            #                                callbacks=[early_stopping, model_checkpoint], verbose=1)

            history = self.autoencoder.fit(self.origin_images, self.combined_labels, epochs=500, batch_size=512,
                                           verbose=1)
            self.autoencoder.save(autoencoder_path)
            self.optimal_epoch = len(history.history['loss'])
            logger.debug('training autoencoder DONE!')
        self.get_border_and_adv()

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
        if self.adv_result is None or self.adv_result.shape[0] == 0:
            self.optimized_adv = np.array([])
            self.L0_befores, self.L0_afters, self.L2_befores, self.L2_afters = [], [], [], []
            return
        else:
            self.optimized_adv = optimize_advs(classifier=self.classifier,
                                               generated_advs=self.adv_result[:4000],
                                               origin_images=self.origin_adv_result[:4000],
                                               target_label=self.target_label,
                                               step=self.step, num_class=10)
        self.end_time = time.time()
        self.optimized_adv = self.adv_result
        self.L0_afters, self.L2_afters = compute_distance(self.optimized_adv, self.origin_adv_result)
        self.L0_befores, self.L2_befores = compute_distance(self.adv_result, self.origin_adv_result)

        if self.adv_result is None:
            return
        if self.adv_result.shape[0] == 0:
            return

        logger.debug(f'adv shape {self.adv_result.shape}')
        logger.debug(f'exe time: ' + str(self.end_time - self.start_time))

    def export_result(self):
        # result = '<=========='
        result = ''
        if self.smooth_adv is not None:
            str_smooth_adv = list(map(str, self.smooth_adv))
            result += '\n'.join(str_smooth_adv)
        if self.adv_result is None or self.adv_result.shape[0] == 0:
            return 0, [], [], [], [], []
        return self.adv_result.shape[0] / float(len(self.origin_images)), self.L0_afters, self.L2_afters, self.smooth_adv, self.L0_befores, self.L2_befores


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
    for weight_index in [0.5]:
        weight_value = weight_index
        # weight_value = weight_index
        for origin_label in range(9, 10):
            for target_position in range(2, 3):
                attacker = AutoencoderBorder(origin_label, np.array(trainX), np.array(trainY), cnn_model,
                                             target_position=target_position, classifier_name=classifier_name,
                                             weight=weight_value, step=step, is_train=False)
                attacker.autoencoder_attack(loss=AE_LOSSES.border_loss)

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

    ranking_type = 'coi'

    # np.savetxt(f'./result/ae4dnn/{classifier_name}_avg_recover_speed_step={step}{ranking_type}.csv', smooth_adv_speed,
    #            delimiter=',')

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
    # f = open('./result/ae_border/' + classifier_name + f'l0_l2_step={step}{ranking_type}.txt', 'w')
    # f.write(l0_l2_txt)
    # f.close()
    # logger.debug('ok')


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

    # a = np.load(os.path.join('result', 'ae_border', 'ae_border_Alexnet_9_7weight=0,1_1000pixels.npy'), allow_pickle=True)
    # b = np.load(os.path.join('result', 'ae_border', 'ae_border_Alexnet_9_7weight=0,1_1000score.npy'), allow_pickle=True)
    # print(a.shape)
    # print(b.shape)
    # print(a[0].shape)

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
