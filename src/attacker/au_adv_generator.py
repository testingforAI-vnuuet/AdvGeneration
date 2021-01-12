"""
Generate adversaries from mnist autoencoder
"""
from __future__ import absolute_import
from tensorflow.keras.datasets import mnist
from attacker.constants import *
from attacker.losses import AE_LOSSES
from data_preprocessing.mnist import mnist_preprocessing
from utility.statistics import *
from matplotlib import pyplot as plt
import keras.losses

logger = MyLogger.getLog()

if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 100
    AUTOENCODER = CLASSIFIER_PATH + '/autoencoder_mnist.h5'
    AE_LOSS = AE_LOSSES.cross_entropy_loss
    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'

    # load autoencoder model
    autoencoder = keras.models.load_model(filepath= AUTOENCODER,
                                          custom_objects={'loss': AE_LOSS})
    logger.debug("Type: %s", type(autoencoder))
    autoencoder.summary()

    # load cnn model
    cnn = keras.models.load_model(filepath=ATTACKED_CNN_MODEL)
    logger.debug("Type: %s", type(cnn))
    cnn.summary()

    # try with some samples on the training set
    (trainX, trainY), (testX, testY) = mnist.load_data()
    pre_mnist = mnist_preprocessing(trainX, trainY, testX, testY, START_SEED, END_SEED, None)
    trainX, trainY, testX, testY = pre_mnist.get_preprocess_data()

    nCol = 6
    # nRow = np.round((END_SEED - START_SEED + 1) / nCol * 2).astype(np.int32)
    nRow = 10
    f = plt.figure()
    pos = 1
    for index in range(START_SEED, END_SEED + 1):
        # logger.debug(index)
        img = trainX[index]
        yLabel = np.argmax(trainY[index])
        original = img.reshape((28, 28))
        trueLabel = np.argmax(trainY[index])

        reconstruction = autoencoder.predict(img.reshape(-1, MNIST_IMG_ROWS, MNIST_IMG_COLS, 1))
        predictionLabel = np.argmax(cnn.predict(reconstruction)[0])

        if trueLabel != predictionLabel:
            logger.debug("Found one")
            # add to plot
            f.add_subplot(nRow, nCol, pos)
            plt.imshow(original, cmap='gray')
            plt.xlabel("origin: " + str(trueLabel))

            f.add_subplot(nRow, nCol, pos + 1)
            plt.imshow(reconstruction.reshape((28, 28)), cmap='gray')
            plt.xlabel("adv: " + str(predictionLabel))

            pos = pos + 2
            plt.subplots_adjust(wspace=1, hspace=1)
    plt.show(block=True)
