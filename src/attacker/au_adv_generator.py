"""
Generate adversaries from mnist autoencoder
"""
from __future__ import absolute_import

import os

import keras.losses
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

from attacker.constants import *
from attacker.losses import AE_LOSSES
from data_preprocessing.mnist import mnist_preprocessing
from utility.statistics import *

logger = MyLogger.getLog()


def generate_adv(autoencoder, loss, cnn_model, trainX, trainY, should_clipping, target, out_image):
    autoencoder = keras.models.load_model(filepath=autoencoder,
                                          custom_objects={'loss': loss})
    logger.debug("Type: %s", type(autoencoder))
    autoencoder.summary()

    # load cnn model
    cnn = keras.models.load_model(filepath=cnn_model)
    logger.debug("Type: %s", type(cnn))
    cnn.summary()

    # try with some samples on the training set
    summary = pd.DataFrame(columns=["index", "l2", "origin", "predict"])
    for index in range(len(trainX)):
        img = trainX[index]
        yLabel = np.argmax(trainY[index])
        original = img.reshape((28, 28))
        trueLabel = np.argmax(trainY[index])

        reconstruction = autoencoder.predict(img.reshape(-1, MNIST_IMG_ROWS, MNIST_IMG_COLS, 1))

        # clipping all pixels to the range of [0..1]
        if should_clipping:
            reconstruction = reconstruction.reshape(MNIST_IMG_ROWS * MNIST_IMG_COLS)
            for index2 in range(len(reconstruction)):
                if reconstruction[index2] > 1:
                    reconstruction[index2] = 1
            reconstruction = reconstruction.reshape(-1, MNIST_IMG_ROWS, MNIST_IMG_COLS, 1)

        # predict
        reconstructedLabel = np.argmax(cnn.predict(reconstruction)[0])

        if trueLabel != target and reconstructedLabel == target:
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
            plt.savefig(out_image + '/' + str(index) + '.png', bbox_inches='tight')

            # export
            logger.debug("Export sample")
            original = original.reshape(-1)
            dataframe_array = pd.DataFrame(original)
            dataframe_array.to_csv(out_image + '/' + str(index) + '_original.csv')

            reconstruction = reconstruction.reshape(-1)
            dataframe_array = pd.DataFrame(reconstruction)
            dataframe_array.to_csv(out_image + '/' + str(index) + '_modified.csv')

            l2 = np.linalg.norm(original - reconstruction)
            summary = summary.append(
                {"index": index, "l2": l2, "origin": trueLabel, "predict": reconstructedLabel,
                 }, ignore_index=True)
    summary.to_csv(out_image + '/summary.csv')
    logger.debug("DONE")


if __name__ == '__main__':
    # create folder to save image
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if not os.path.exists(ROOT + '/data/'):
        os.mkdir(ROOT + '/data/')
    OUT_IMGAGES = ROOT + '/data/mnist/'
    if not os.path.exists(OUT_IMGAGES):
        os.mkdir(OUT_IMGAGES)

    # load autoencoder model
    generate_adv(autoencoder= CLASSIFIER_PATH + '/autoencoder_mnist.h5',
                 loss=AE_LOSSES.cross_entropy_loss,
                 cnn_model= CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5',
                 start_seed=0,
                 end_seed=1000,
                 should_clipping=True,
                 target=7,
                 out_image=OUT_IMGAGES)
