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
from attacker.matrix_attack import get_mnist_output, generate_name_for_model
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *

logger = MyLogger.getLog()


def generate_adv(auto_encoder_path: str,
                 loss,
                 cnn_model_path: str,
                 train_X: np.ndarray,
                 train_Y: np.ndarray,
                 should_clipping: bool,
                 target: int,
                 out_image: str):
    auto_encoder = keras.models.load_model(filepath=auto_encoder_path,
                                           custom_objects={'loss': loss})
    logger.debug("Type: %s", type(auto_encoder))
    auto_encoder.summary()

    # load cnn model
    cnn = keras.models.load_model(filepath=cnn_model_path)
    logger.debug("Type: %s", type(cnn))
    cnn.summary()

    # try with some samples on the training set
    summary = pd.DataFrame(columns=["index", "l2", "origin", "predict"])
    for index in range(len(train_X)):
        img = train_X[index]
        yLabel = np.argmax(train_Y[index])
        original = img.reshape((28, 28))
        trueLabel = np.argmax(train_Y[index])

        reconstruction = auto_encoder.predict(img.reshape(-1, MNIST_IMG_ROWS, MNIST_IMG_COLS, 1))

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
    START_ATTACK_SEED, END_ATTACK_SEED = 10000, 12000
    SOURCE_LABEL = 0
    TARGET = 1
    AE_LOSS = AE_LOSSES.cross_entropy_loss
    CNN_MODEL_PATH = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    AE_MODEL = '/Users/ducanhnguyen/Documents/PycharmProjects/AdvGeneration/data/mnist/model/0_to_1.h5'

    # load dataset
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    pre_mnist = MnistPreprocessing(train_X, train_Y, test_X, test_Y, START_ATTACK_SEED, END_ATTACK_SEED, TARGET)
    train_X, train_Y, test_X, test_Y = pre_mnist.preprocess_data()
    countSamples(probability_vector=train_Y, n_class=MNIST_NUM_CLASSES)

    # create folder to save image
    get_mnist_output()

    # load autoencoder model
    OUT_IMGAGES = get_mnist_output() + '/' + generate_name_for_model(SOURCE_LABEL, TARGET)
    if not os.path.exists(OUT_IMGAGES):
        os.mkdir(OUT_IMGAGES)

    #
    removed_labels = []
    for i in range(MNIST_NUM_CLASSES):
        if i != SOURCE_LABEL:
            removed_labels.append(i)
    #
    (train_X2, train_Y2), (test_X2, test_Y2) = mnist.load_data()
    pre_mnist2 = MnistPreprocessing(train_X2, train_Y2, test_X2, test_Y2, START_ATTACK_SEED,
                                    END_ATTACK_SEED,
                                    removed_labels=removed_labels)
    train_X2, train_Y2, _, _ = pre_mnist2.preprocess_data()
    countSamples(probability_vector=train_Y2, n_class=MNIST_NUM_CLASSES)

    generate_adv(auto_encoder_path=AE_MODEL,
                 loss=AE_LOSS,
                 cnn_model_path=CNN_MODEL_PATH,
                 train_X=train_X2,
                 train_Y=train_Y2,
                 should_clipping=True,
                 target=TARGET,
                 out_image=OUT_IMGAGES)
