import math

from tensorflow.python.keras import Sequential

from attacker.constants import CLASSIFIER_PATH, MNIST_NUM_CLASSES, MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL
from tensorflow.keras.datasets import mnist

from classifier.cnnmodel import MNIST
from data_preprocessing.mnist import mnist_preprocessing
from utility.mylogger import MyLogger
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

logger = MyLogger.getLog()

if __name__ == '__main__':
    classifier = MNIST().get_architecture()

    if isinstance(classifier, Sequential):
        (trainX, trainY), (testX, testY) = mnist.load_data()
        pre_mnist = mnist_preprocessing(trainX, trainY, testX, testY, None, None, None)
        trainX, trainY, testX, testY = pre_mnist.get_preprocess_data()

        #
        # loss_object = keras.losses.CategoricalCrossentropy()
        # input_image = trainX[0]
        # input = tf.convert_to_tensor([input_image])
        # with tf.GradientTape() as tape:
        #     tape.watch(input)
        #     prediction = classifier(input)
        #     loss = keras.losses.CategoricalCrossentropy()\
        #         .__call__(y_true=trainY[0].reshape(1, 10), y_pred=prediction)
        # gradient = tape.gradient(loss, input)
        # print(gradient)

        index = 10
        input_image = trainX[index]
        logger.debug("input shape: " + str(input_image.shape))

        loss_object = keras.losses.CategoricalCrossentropy()
        input = tf.convert_to_tensor([input_image])
        trueLabel = np.argmax(trainY[index])
        logger.debug("True label: " + str(trueLabel))

        # gradient
        with tf.GradientTape() as tape:
            tape.watch(input)
            prediction_at_target_neuron = classifier(input)[0][trueLabel]
        gradient = tape.gradient(prediction_at_target_neuron, input)
        gradient = gradient.numpy()

        gradient = gradient.reshape(MNIST_IMG_ROWS * MNIST_IMG_COLS)

        for i in range(0, 10):
            maxIndex = np.argmax(gradient)
            logger.debug("max index = " + str(maxIndex))
            rowIndex = math.floor(maxIndex / MNIST_IMG_COLS)
            colIndex = maxIndex - rowIndex * MNIST_IMG_COLS
            logger.debug("(" + str(rowIndex) + ", " + str(colIndex) + ")")
            input_image[rowIndex][colIndex][0] = 2
            gradient[maxIndex] = 0

        plt.imshow(input_image, cmap='gray')
        plt.show()
