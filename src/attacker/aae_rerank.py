from __future__ import absolute_import

import os
import threading
import time

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from attacker.autoencoder import MnistAutoEncoder
from attacker.constants import *
from attacker.losses import *
from attacker.mnist_utils import reject_outliers, L0, L2
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *

logger = MyLogger.getLog()

tf.config.experimental_run_functions_eagerly(True)

pretrained_model_name = ['Alexnet', 'Lenet', 'vgg13', 'vgg16']


class Auto_encoder_rerank:

    def __init__(self, trainX, trainY, origin_label, target_position, classifier, weight, classifier_name='noname'):
        """

        :param target: target class in attack
        """
        self.method_name = 'atn'
        self.start_time = time.time()
        self.origin_label = origin_label
        self.classifier = classifier
        self.trainX = trainX
        self.trainY = trainY
        self.target_position = target_position
        self.weight = weight

        self.origin_images, self.origin_labels = filter_by_label(self.origin_label, self.trainX, self.trainY)

        self.origin_images = np.array(self.origin_images[:2000])
        self.origin_labels = np.array(self.origin_labels[:2000])

        self.target_label = label_ranking(self.origin_images, self.classifier)[-1 * self.target_position]
        self.target_vector = tf.keras.utils.to_categorical(self.target_label, MNIST_NUM_CLASSES, dtype='float32')

        self.autoencoder = None

        self.file_shared_name = self.method_name + '_' + classifier_name + f'_{origin_label}_{self.target_label}' + 'weight' + str(
            self.weight).replace('.', ',')

        self.autoencoder_file_name = self.file_shared_name + 'autoencoder' + '.h5'
        self.optimal_epoch = 0
        self.generated_candidates = None
        self.adv_result = None
        self.origin_adv_result = None
        self.end_time = None
        self.adv_result_file_path = self.file_shared_name + '_adv_result' + '.npy'
        self.origin_adv_result_file_path = self.file_shared_name + '_origin_adv_result' + '.npy'
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
        self.adv_result, self.num_avg_redundant_pixels = restore_redundant_mnist_pixels(self.classifier, self.adv_result, self.origin_adv_result,
                                                         self.target_label)
        np.save(os.path.join(SAVED_NPY_PATH, self.method_name, self.adv_result_file_path), self.adv_result)
        np.save(os.path.join(SAVED_NPY_PATH, self.method_name, self.origin_adv_result_file_path),
                self.origin_adv_result)
        self.end_time = time.time()

    def export_result(self):
        result = '<=========='
        result += '\norigin=' + str(self.origin_label) + ',target=' + str(self.target_label) + '\n'
        result += '\n\tweihjt=' + str(self.weight)
        result += '\n\t#adv=' + str(self.adv_result.shape[0])
        result += '\n\t#optimal_epoch=' + str(self.optimal_epoch)
        result += '\n\t#avg_redundant_pixels=' + str(self.num_avg_redundant_pixels)

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
        l2_l0_best = l0[best_l0_index]

        plot_images(origin_image_worst_l0, origin_image_best_l0, gen_image_worst_l0, gen_image_best_l0, l2_l0_worst,
                    l2_l0_best, l0_worst, l0_best, path_l0, self.classifier, worst_l0_index, best_l0_index)

    # def get_architecture(self, input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)):
    #     input_img = keras.layers.Input(shape=input_shape)
    #     x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    #     x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    #     x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #     encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    #
    #     x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    #     x = keras.layers.UpSampling2D((2, 2))(x)
    #     x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #     x = keras.layers.UpSampling2D((2, 2))(x)
    #     decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    #     self.auto_encoder = keras.models.Model(input_img, decoded)
    #     self.auto_encoder.summary()

    # def compile(self, classifier, loss):
    #     if not isinstance(self.auto_encoder, keras.models.Model):
    #         self.get_architecture()
    #
    #     adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #     self.auto_encoder.compile(optimizer=adam,
    #                               loss=loss(classifier, self.target_label_onehot, weight=0.005, alpha=1.5))
    #     self.is_compiled = True
    #
    # def fit(self, epochs, batch_size):
    #     """
    #     :param epochs: epochs
    #     :param batch_size: batch size
    #     :return: null
    #     """
    #     if not self.is_compiled:
    #         raise ValueError("Compile the autoencoder first")
    #     self.auto_encoder.train(self.get_seed_images(), self.get_seed_images(), epochs=epochs, batch_size=batch_size)
    #     self.plot(self.auto_encoder.history)

    # def plot(self, history):
    #     plt.plot(history.history['loss'])
    #     plt.title('Loss')
    #     plt.ylabel('loss')
    #     plt.xlabel('epoch')
    #     # plt.legend(['train', 'test'], loc='upper left')
    #     plt.show()


def run_thread(classifier_name, trainX, trainY):
    cnn_model = tf.keras.models.load_model(PRETRAIN_CLASSIFIER_PATH + '/' + classifier_name + '.h5')
    result_txt = classifier_name + '\n'

    origin_label = 9
    target_position = 2

    for weight_index in range(0, 11):
        weight_value = weight_index * 0.1
        attacker = Auto_encoder_rerank(trainX=trainX, trainY=trainY, origin_label=origin_label,
                                       target_position=target_position, classifier=cnn_model,
                                       classifier_name=classifier_name, weight=weight_value)
        attacker.autoencoder_attack(loss=AE_LOSSES.re_rank_loss)
        res_txt, _ = attacker.export_result()
        attacker.save_images()
        result_txt += res_txt

    f = open('./result/atn/' + classifier_name + str(origin_label) + '.txt', 'w')
    f.write(result_txt)
    f.close()


class MyThread(threading.Thread):
    def __init__(self, classifier_name, trainX, trainY):
        super(MyThread, self).__init__()
        self.classifier_name = classifier_name
        self.trainX = trainX
        self.trainY = trainY

    def run(self):
        run_thread(self.classifier_name, self.trainX, self.trainY)


if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 1000
    TARGET = 7
    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    AE_MODEL = CLASSIFIER_PATH + '/autoencoder_rerank_mnist.h5'
    AE_LOSS = AE_LOSSES.re_rank_loss

    # load dataset
    # (trainX, trainY), (testX, testY) = mnist.load_data()
    # pre_mnist = MnistPreprocessing(trainX, trainY, testX, testY, START_SEED, END_SEED, TARGET)
    # trainX, trainY, testX, testY = pre_mnist.preprocess_data()
    # countSamples(probability_vector=trainY, n_class=MNIST_NUM_CLASSES)
    #
    # # train an autoencoder
    # classifier = keras.models.load_model(ATTACKED_CNN_MODEL)
    # auto_encoder = Auto_encoder_rerank(train_data=trainX, target=TARGET, nClasses=MNIST_NUM_CLASSES)
    # auto_encoder.compile(classifier=classifier, loss=AE_LOSS)
    # auto_encoder.fit(epochs=300, batch_size=512)
    #
    # logger.debug("Export the trained auto-encoder to file")
    # auto_encoder.auto_encoder.save(AE_MODEL)

    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

    trainX, trainY = MnistPreprocessing.quick_preprocess_data(trainX, trainY, num_classes=MNIST_NUM_CLASSES,
                                                              rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                              chl=MNIST_IMG_CHL)
    testX, testY = MnistPreprocessing.quick_preprocess_data(testX, testY, num_classes=MNIST_NUM_CLASSES,
                                                            rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                            chl=MNIST_IMG_CHL)

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
