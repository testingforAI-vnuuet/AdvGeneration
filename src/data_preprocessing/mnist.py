import numpy as np
from .constants import *


class mnist_preprocessing:

    def __init__(self, trainX, trainY, testX, testY, start, end, target):
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
        self.target = target

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

    def preprocess_data(self, trainX, trainY, is_processed):
        """
        preprocess mnist data
        :return: null
        """
        trainX = trainX.reshape((-1, MNIST_IMG_ROWS, MNIST_IMG_ROWS, MNIST_IMG_CHL))
        if not is_processed:
            trainX = trainX / 255.0
            trainY = self.convert_to_onehot_vector(trainY)
            is_processed = True
        return trainX, trainY, is_processed

    def removeLabel(self, trainX, trainY, target):
        removedColIdx = 0 # first dimension
        for index in range(len(trainX) - 1, -1, -1):
            if trainY[index] == target:
                trainX = np.delete(trainX, index, removedColIdx)
                trainY = np.delete(trainY, index)
        return trainX, trainY

    def cutoff(self, start, end, trainX, trainY):
        return trainX[start: end + 1], trainY[start: end + 1]

    def get_preprocess_data(self):
        """
        Get mnist data after preprocessing
        :return: preprocessed data
        """
        assert self.trainY is not None
        assert self.trainX is not None

        if self.start is not None and self.end is not None:
            self.trainX, self.trainY = self.cutoff(self.start, self.end, self.trainX, self.trainY)
        if self.target is not None:
            self.trainX, self.trainY = self.removeLabel(self.trainX, self.trainY, self.target)
        self.trainX, self.trainY, self.is_processed = \
            self.preprocess_data(self.trainX, self.trainY, self.is_processed)
        return self.trainX, self.trainY, self.testX, self.testY
