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

pretrained_model_name = ['Alexnet', 'Lenet_v2', 'vgg13', 'vgg16']


def combined_function(set1, set2, set3):
    return np.array([list(combined) for combined in zip(set1, set2, set3)])


class PrimaryAutoencoderBorder:
    def __init__(self, origin_label, trainX, trainY, classifier, weight, target_position=2, classifier_name='noname',
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
        self.method_name = 'ae_border'

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
            self.autoencoder.compile(optimizer=adam,
                                     loss=loss(self.classifier, self.target_vector, self.origin_images, self.weight))
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
                                     loss=loss(self.classifier, self.target_vector, self.origin_images, self.weight))
            # self.autoencoder.compile(optimizer=adam, loss=tf.keras.losses.binary_crossentropy)
            early_stopping = EarlyStopping(monitor='loss', verbose=0, mode='min')
            model_checkpoint = ModelCheckpoint(autoencoder_path,
                                               save_best_only=True, monitor='loss',
                                               mode='min')

            history = self.autoencoder.fit(self.origin_images, self.combined_labels, epochs=500, batch_size=512,
                                           callbacks=[early_stopping, model_checkpoint], verbose=1)
            self.optimal_epoch = len(history.history['loss'])
            logger.debug('training autoencoder DONE!')

    def get_border_and_adv(self):

        # self.generated_borders = []
        # self.generated_candidates = []
        logger.debug('getting advs')
        candidate_generated_borders = self.autoencoder.predict(
            self.origin_images[:self.num_images]) * self.border_origin_images[:self.num_images]

        self.generated_candidates = np.clip(candidate_generated_borders + self.internal_origin_images[:self.num_images],
                                            0, 1)
        self.generated_borders = np.array(candidate_generated_borders)
        self.adv_result, _, self.origin_adv_result, _ = filter_candidate_adv(self.origin_images[:self.num_images],
                                                                             self.generated_candidates,
                                                                             self.target_label,
                                                                             cnn_model=self.classifier)

        # self.smooth_adv = smooth_adv_border_V2(self.classifier, self.adv_result, self.origin_adv_result,
        #                                        get_border(self.origin_adv_result), self.target_label)
        self.end_time = time.time()
        logger.debug('get advs DONE!')

    def export_result(self):
        result = '<=========='
        # str_smooth_adv = list(map(str, self.smooth_adv))
        # result += '\n' + '\n'.join(str_smooth_adv)

        # result += '\norigin=' + str(self.origin_label) + ',target=' + str(self.target_label) + '\n'
        # result += '\n\t#adv=' + str(self.adv_result.shape[0])
        # result += '\n\t#optimal_epoch=' + str(self.optimal_epoch)
        # l0 = np.array([L0(gen, test) for gen, test in zip(self.adv_result, self.origin_adv_result)])
        # l0 = reject_outliers(l0)
        #
        # if l0.shape[0] != 0:
        #     result += '\n\tl0=' + str(min(l0)) + '/' + str(max(l0)) + '/' + str(np.average(l0))
        # else:
        #     result += '\n\tl0=None'
        #
        # l2 = np.array([L2(gen, test) for gen, test in zip(self.adv_result, self.origin_adv_result)])
        # l2 = reject_outliers(l2)
        #
        # if l2.shape[0] != 0:
        #     result += '\n\tl2=' + str(round(min(l2), 2)) + '/' + str(round(max(l2), 2)) + '/' + str(
        #         round(np.average(l2), 2))
        # else:
        #     result += '\n\tl2=None'
        # result += '\n\ttime=' + str(self.end_time - self.start_time) + ' s'
        # result += '\n==========>\n'
        # return result, self.end_time - self.start_time
        # f = open(os.path.join('result', self.method_name, self.file_shared_name + '.txt', ), 'w')
        # f.write(result)
        # f.close()
        #
        # return result, self.end_time - self.start_time
        return self.adv_result.shape[0] / float(self.num_images)

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

        path_l2 = SAVED_IMAGE_SAMPLE_PATH + '/epsilon1/l2/' + 'figure_' + self.file_shared_name + '.png'
        path_l0 = SAVED_IMAGE_SAMPLE_PATH + '/epsilon1/l0/' + 'figure_' + self.file_shared_name + '.png'

        # show for l2
        origin_image_worst_l2 = self.origin_adv_result[worst_l2_index]
        origin_image_best_l2 = self.origin_adv_result[best_l2_index]

        gen_image_worst_l2 = self.adv_result[worst_l2_index]
        gen_image_best_l2 = self.adv_result[best_l2_index]

        l2_worst = l2[worst_l2_index]
        l2_best = l2[best_l2_index]

        l0_l2_worst = l0[worst_l2_index]
        l0_l2_best = l0[best_l2_index]

        # plot_images(origin_image_worst_l2, origin_image_best_l2, gen_image_worst_l2, gen_image_best_l2, l2_worst,
        # l2_best, l0_l2_worst, l0_l2_best, path_l2, self.classifier, worst_l2_index, worst_l0_index)

        # show for l0
        origin_image_worst_l0 = self.origin_adv_result[worst_l0_index]
        origin_image_best_l0 = self.origin_adv_result[best_l0_index]

        gen_image_worst_l0 = self.adv_result[worst_l0_index]
        gen_image_best_l0 = self.adv_result[best_l0_index]

        l0_worst = l0[worst_l0_index]
        l0_best = l0[best_l0_index]

        l2_l0_worst = l2[worst_l0_index]
        l2_l0_best = l0[best_l0_index]

        plot_images(origin_image_worst_l0, origin_image_best_l0, gen_image_worst_l0, gen_image_best_l0, l2_l0_worst,
                    l2_l0_best, l0_worst, l0_best, path_l0, self.classifier, worst_l0_index, best_l0_index)


def run_thread(classifier_name, trainX, trainY):
    logger.debug("\n=======================================================")
    logger.debug('processing model: ' + classifier_name)
    cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    result_txt = classifier_name + '\n'
    # AE_LOSS = AE_LOSSES.border_loss
    for origin_label in range(9, 10):
        # exe_time_sum = 0
        for target_position in range(2, 3):
            for weight_index in range(0, 11):
                weight_value = weight_index * 0.1
                attacker = PrimaryAutoencoderBorder(origin_label, np.array(trainX), np.array(trainY), cnn_model,
                                                    target_position=target_position, classifier_name=classifier_name,
                                                    weight=weight_value)
                attacker.autoencoder_attack(loss=AE_LOSSES.border_loss)
                attacker.get_border_and_adv()
                attacker.export_result()
                del attacker
                # attacker.save_images()
                # res_txt, exe_time = attacker.export_result()
        #         result_txt += res_txt
        #         exe_time_sum += exe_time
        # f = open('./result/' + classifier_name + str(origin_label) + '.txt', 'w')
        # result_txt += '\n average_time = ' + str(exe_time_sum / 9.) + '\n'
        # f.write(result_txt)
        # f.close()
        logger.debug('processing model: ' + classifier_name + ' DONE!')
        logger.debug("=======================++++============================")


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
                attacker = PrimaryAutoencoderBorder(origin_label, np.array(trainX), np.array(trainY), cnn_model,
                                                    target_position=target_position, classifier_name=classifier_name,
                                                    weight=weight_value)
                attacker.autoencoder_attack(loss=AE_LOSSES.border_loss)
                attacker.get_border_and_adv()
                weight_result_i_j.append(attacker.export_result())
                del attacker
            weight_result_i.append(weight_result_i_j)
        weight_result_i = np.average(weight_result_i, axis=0)
        weight_result.append(weight_result_i)

    weight_result = np.array(weight_result)
    s = np.array2string(weight_result, separator=' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    f = open('./result/primary_ae_border/' + classifier_name + '.txt', 'w')
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
    # thread1 = MyThread(pretrained_model_name[0], trainX, trainY)
    thread2 = MyThread(pretrained_model_name[1], trainX, trainY)
    # thread3 = MyThread(pretrained_model_name[2], trainX, trainY)
    # thread4 = MyThread(pretrained_model_name[3], trainX, trainY)

    # thread1.start()
    thread2.start()
    # thread3.start()
    # thread4.start()

    # thread1.join()
    thread2.join()
    # thread3.join()
    # thread4.join()

    logger.debug('Exiting Main Thread')
    logger.debug('robustness testing DONE !')
