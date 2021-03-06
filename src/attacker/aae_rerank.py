from __future__ import absolute_import

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from attacker.constants import *
from attacker.losses import *
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *

logger = MyLogger.getLog()

tf.config.experimental_run_functions_eagerly(True)
class Auto_encoder_rerank:

    def __init__(self, train_data, target, nClasses):
        """

        :param target: target class in attack
        """
        self.auto_encoder = None
        self.target = target
        self.target_label_onehot = keras.utils.to_categorical(target, nClasses, dtype='float32')
        self.train_data = train_data
        self.is_compiled = False

    def get_seed_images(self):
        """
        :return: seed images
        """
        return self.train_data

    def get_architecture(self, input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)):
        input_img = keras.layers.Input(shape=input_shape)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        self.auto_encoder = keras.models.Model(input_img, decoded)
        self.auto_encoder.summary()

    def compile(self, classifier, loss):
        if not isinstance(self.auto_encoder, keras.models.Model):
            self.get_architecture()

        adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.auto_encoder.compile(optimizer=adam,
                                  loss=loss(classifier, self.target_label_onehot, weight=0.005, alpha=1.5))
        self.is_compiled = True

    def fit(self, epochs, batch_size):
        """
        :param epochs: epochs
        :param batch_size: batch size
        :return: null
        """
        if not self.is_compiled:
            raise ValueError("Compile the autoencoder first")
        self.auto_encoder.train(self.get_seed_images(), self.get_seed_images(), epochs=epochs, batch_size=batch_size)
        self.plot(self.auto_encoder.history)

    def plot(self, history):
        plt.plot(history.history['loss'])
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 1000
    TARGET = 7
    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    AE_MODEL = CLASSIFIER_PATH + '/autoencoder_rerank_mnist.h5'
    AE_LOSS = AE_LOSSES.re_rank_loss

    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    pre_mnist = MnistPreprocessing(trainX, trainY, testX, testY, START_SEED, END_SEED, TARGET)
    trainX, trainY, testX, testY = pre_mnist.preprocess_data()
    countSamples(probability_vector=trainY, n_class=MNIST_NUM_CLASSES)

    # train an autoencoder
    classifier = keras.models.load_model(ATTACKED_CNN_MODEL)
    auto_encoder = Auto_encoder_rerank(train_data=trainX, target=TARGET, nClasses=MNIST_NUM_CLASSES)
    auto_encoder.compile(classifier=classifier, loss=AE_LOSS)
    auto_encoder.fit(epochs=300, batch_size=512)

    logger.debug("Export the trained auto-encoder to file")
    auto_encoder.auto_encoder.save(AE_MODEL)
