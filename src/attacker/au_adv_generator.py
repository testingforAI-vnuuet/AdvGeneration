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
import pandas as pd
import math
import os
logger = MyLogger.getLog()

if __name__ == '__main__':
    TARGET = 7
    START_SEED = 0
    END_SEED = 1000
    AUTOENCODER = CLASSIFIER_PATH + '/autoencoder_mnist_relu_same.h5'
    AE_LOSS = AE_LOSSES.cross_entropy_loss
    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    SHOULD_CLIPPING = False

    # create folder to save image
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if not os.path.exists(ROOT + '/data/'):
        os.mkdir(ROOT + '/data/')
    OUT_IMGAGES = ROOT + '/data/mnist/'
    if not os.path.exists(OUT_IMGAGES):
        os.mkdir(OUT_IMGAGES)

    # load autoencoder model
    autoencoder = keras.models.load_model(filepath=AUTOENCODER,
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

    summary = pd.DataFrame(columns=["index", "l2", "origin", "predict"])
    for index in range(START_SEED, END_SEED + 1):
        img = trainX[index]
        yLabel = np.argmax(trainY[index])
        original = img.reshape((28, 28))
        trueLabel = np.argmax(trainY[index])

        reconstruction = autoencoder.predict(img.reshape(-1, MNIST_IMG_ROWS, MNIST_IMG_COLS, 1))

        # clipping all pixels to the range of [0..1]
        if SHOULD_CLIPPING:
            reconstruction = reconstruction.reshape(MNIST_IMG_ROWS* MNIST_IMG_COLS)
            for index2 in range(len(reconstruction)):
                if reconstruction[index2] > 1:
                    reconstruction[index2] = 1
            reconstruction = reconstruction.reshape(-1, MNIST_IMG_ROWS, MNIST_IMG_COLS, 1)

        # predict
        reconstructedLabel = np.argmax(cnn.predict(reconstruction)[0])

        if trueLabel != TARGET and reconstructedLabel == TARGET:
            logger.debug("Found index " + str(index))

            # full figure
            logger.debug("Export image")
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(original, cmap='gray')
            plt.xlabel("origin: " + str(trueLabel))

            f.add_subplot(1, 2, 2)
            plt.imshow(reconstruction.reshape((28, 28)), cmap='gray')
            plt.xlabel("adv: " + str(reconstructedLabel))

            plt.subplots_adjust(wspace=1, hspace=1)
            plt.savefig(OUT_IMGAGES + '/' + str(index) + '.png', bbox_inches='tight')

            # export
            logger.debug("Export sample")
            original = original.reshape(-1)
            dataframe_array = pd.DataFrame(original)
            dataframe_array.to_csv(OUT_IMGAGES + '/' + str(index) + '_original.csv')

            reconstruction = reconstruction.reshape(-1)
            dataframe_array = pd.DataFrame(reconstruction)
            dataframe_array.to_csv(OUT_IMGAGES + '/' + str(index) + '_modified.csv')

            l2 = math.dist(original, reconstruction)
            summary = summary.append(
                {"index": index, "l2": l2, "origin": trueLabel, "predict": reconstructedLabel,
                 }, ignore_index=True)
    summary.to_csv(OUT_IMGAGES + '/summary.csv')
    logger.debug("DONE")
