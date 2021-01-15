from tensorflow import keras
from tensorflow.python.keras import Sequential
from attacker.constants import MNIST_IMG_ROWS, MNIST_IMG_COLS, CLASSIFIER_PATH, MNIST_IMG_CHL
from tensorflow.keras.datasets import mnist
from data_preprocessing.mnist import mnist_preprocessing
from utility.mylogger import MyLogger
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import enum

logger = MyLogger.getLog()


class RANKING_ALGORITHM(enum.Enum):
    ABS = 1
    COI = 2
    CO = 3


class PIXEL_RANKER:
    def __init__(self):
        return

    @staticmethod
    def compute_gradient_wrt_features(input: tf.Tensor,
                                      target_neuron: int,
                                      classifier: tf.keras.Sequential):
        """Compute gradient wrt features.

        Args:
            input: a tensor (shape = `[1, height, width, channel`])
            target_neuron: the index of the neuron on the output layer needed to be differentiated
            classifier: a sequential model
        Returns:
            gradient: ndarray (shape = `[height, width, channel`])
        """
        with tf.GradientTape() as tape:
            tape.watch(input)
            prediction_at_target_neuron = classifier(input)[0][target_neuron]
        gradient = tape.gradient(prediction_at_target_neuron, input)
        gradient = gradient.numpy()[0]
        return gradient

    @staticmethod
    def find_important_pixels(gradient: np.ndarray,
                              input_image: np.ndarray,
                              n_rows: int, n_cols: int, n_channels: int, n_important_features: int,
                              algorithm: enum.Enum):
        """Apply ABS algorithm to find the most important features.

        Args:
            gradient: the same shape as input (shape = `[height, width, channel`])
            n_rows: a positive number
            n_cols: a positive number
            n_channels: a positive number
            n_important_features: a positive number
            input_image: shape = `[height, width, channel`]
            algorithm:
        Returns:
            positions: ndarray (shape=`[index_row, index_col, index_channel`])
        """
        input_image = input_image.copy()  # avoid modifying on the original one
        gradient = gradient.copy()
        positions = np.ndarray(shape=(1, 3), dtype=int)
        foundSet = {}

        # find the position of the highest value in the gradient
        for idx in range(0, n_important_features):
            max_row = max_col = max_channel = None
            max_value = -99999999
            for rdx in range(0, n_rows):
                for cdx in range(0, n_cols):
                    for chdx in range(0, n_channels):
                        changed = False
                        hash = rdx + cdx * 1.3243 + chdx * 1.53454
                        if hash in foundSet:
                            continue
                        if algorithm == RANKING_ALGORITHM.ABS:
                            grad = gradient[rdx, cdx, chdx]
                            if np.abs(grad) > max_value:
                                max_value = np.abs(grad)
                                changed = True
                        elif algorithm == RANKING_ALGORITHM.CO:
                            grad = gradient[rdx, cdx, chdx]
                            if grad > max_value:
                                max_value = grad
                                changed = True
                        elif algorithm == RANKING_ALGORITHM.COI:
                            feature_value = input_image[rdx, cdx, chdx]
                            grad = gradient[rdx, cdx, chdx]
                            if grad * feature_value > max_value:
                                max_value = grad * feature_value
                                changed = True
                        if changed:
                            foundSet[hash] = 0
                            max_row = rdx
                            max_col = cdx
                            max_channel = chdx

            # after iterating all pixels
            if max_row is not None:
                positions = np.append(positions, [[max_row, max_col, max_channel]], axis=0)
                input_image[max_row, max_col, max_channel] = np.max(gradient) + 2

        positions = np.delete(positions, 0, axis=0)
        return positions

    @staticmethod
    def display(positions, input_image):
        input_image = input_image.copy()
        max = np.max(input_image)
        ROW_IDX = 0
        COL_IDX = 1
        CHANNEL_INDEX = 2
        for idx in range(0, positions.shape[0]):
            row = positions[idx, ROW_IDX]
            col = positions[idx, COL_IDX]
            channel = positions[idx, CHANNEL_INDEX]
            input_image[row, col, channel] = max + 2
        plt.imshow(input_image, cmap='gray')
        plt.title("Most important features are highlighted")
        plt.show()


if __name__ == '__main__':
    classifier = keras.models.load_model(CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5')
    INDEX = 10

    if isinstance(classifier, Sequential):
        # get a seed
        (trainX, trainY), (testX, testY) = mnist.load_data()
        pre_mnist = mnist_preprocessing(trainX, trainY, testX, testY, None, None, None)
        trainX, trainY, testX, testY = pre_mnist.get_preprocess_data()

        input_image = trainX[INDEX]
        logger.debug("Input shape: " + str(input_image.shape))
        trueLabel = np.argmax(trainY[INDEX])
        logger.debug("True label: " + str(trueLabel))

        # gradient the neuron output corresponding to the true label w.r.t features
        gradient = PIXEL_RANKER.compute_gradient_wrt_features(
            input=tf.convert_to_tensor([input_image]),
            target_neuron=trueLabel,
            classifier=classifier)

        positions = PIXEL_RANKER.find_important_pixels(
            gradient=gradient,
            n_rows=MNIST_IMG_ROWS,
            n_cols=MNIST_IMG_COLS,
            n_channels=MNIST_IMG_CHL,
            input_image=input_image,
            n_important_features=10,
            algorithm=RANKING_ALGORITHM.CO
        )
        logger.debug("Important positions: " + str(positions))
        PIXEL_RANKER.display(positions, input_image.copy())
