import numpy as np
from .constants import *


class mnist_preprocessing:

    def __init__(self, trainX, trainY, testX, testY):
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

    def preprocess_data(self):
        """
        preprocess mnist data
        :return: null
        """
        self.trainX = self.trainX.reshape((-1, MNIST_IMG_ROWS, MNIST_IMG_ROWS, MNIST_IMG_CHL))
        if not self.is_processed:
            self.trainX = self.trainX / 255.0
            self.trainY = self.convert_to_onehot_vector(self.trainY)
            self.is_processed = True

    def get_preprocess_data(self):
        """
        Get mnist data after preprocessing
        :return: preprocessed data
        """
        self.preprocess_data()
        return self.trainX, self.trainY, self.testX, self.testY
