import numpy as np
import tensorflow as tf

from .constants import *


class MnistPreprocessing:

    def __init__(self, trainX, trainY, testX, testY, start, end, removed_labels):
        """
        Initialize training and testing data
        All input in numpy array
        :param trainX: array of training image data
        :param trainY: array of training label data
        :param testX:  array of testing image data
        :param testY: array of testing label data
        """
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.is_processed = False
        self.start = start
        self.end = end
        self.removed_labels = removed_labels

    @staticmethod
    def convert_to_onehot_vector(data):
        """
        Convert array of label to array of one-hot vector
        :param data: input data
        :return: array of one-hot vector
        """
        # print(int(np.max(data)))
        # print(data.shape[0])
        data = np.array(data, dtype='float32')
        result = np.zeros(shape=(data.shape[0], int(np.max(data)) + 1))
        result[np.arange(data.shape[0]), np.array(data, dtype='int32')] = 1
        return result

    def remove_label_and_update(self, target):
        self.trainX, self.trainY = self.remove_label(self.trainX, self.trainY, target)

    def remove_label(self, trainX: np.ndarray, trainY: np.ndarray, target: int):
        removedColIdx = 0  # first dimension

        obj = []
        for index in range(len(trainX) - 1, -1, -1):
            if trainY[index] == target:
                obj.append(index)

        trainX = np.delete(trainX, obj, removedColIdx)
        trainY = np.delete(trainY, obj)
        return trainX, trainY

    def cutoff(self, start, end, trainX, trainY):
        return trainX[start: end + 1], trainY[start: end + 1]

    def preprocess_data(self):
        """Get mnist data after preprocessing
        """
        assert self.trainY is not None
        assert self.trainX is not None

        # cut off samples
        if self.start is not None and self.end is not None:
            self.trainX, self.trainY = self.cutoff(self.start, self.end, self.trainX, self.trainY)

        # remove labels
        if self.removed_labels is not None:
            if isinstance(self.removed_labels, int):
                self.trainX, self.trainY = self.remove_label(self.trainX, self.trainY, self.removed_labels)
            else:
                for remove in self.removed_labels:
                    self.trainX, self.trainY = self.remove_label(self.trainX, self.trainY, remove)

        # normalize
        self.trainX = self.trainX.reshape((-1, MNIST_IMG_ROWS, MNIST_IMG_ROWS, MNIST_IMG_CHL))
        self.trainX = self.trainX / 255.0

        self.trainY = self.convert_to_onehot_vector(self.trainY)

        return self.trainX, self.trainY, self.testX, self.testY

    @staticmethod
    def quick_preprocess_data(images, label, num_classes, rows, cols, chl):

        images = images.astype('float32') / 255.
        return images.reshape((len(images), rows, cols, chl)), \
               tf.keras.utils.to_categorical(label, num_classes)

# custom
