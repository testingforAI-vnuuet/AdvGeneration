import math

from tensorflow.python.keras import Sequential

from attacker.constants import MNIST_IMG_ROWS, MNIST_IMG_COLS
from tensorflow.keras.datasets import mnist

from classifier.cnnmodel import MNIST
from data_preprocessing.mnist import mnist_preprocessing
from utility.mylogger import MyLogger
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

logger = MyLogger.getLog()

if __name__ == '__main__':
    classifier = MNIST().get_architecture()

    if isinstance(classifier, Sequential):
        # get a seed
        (trainX, trainY), (testX, testY) = mnist.load_data()
        pre_mnist = mnist_preprocessing(trainX, trainY, testX, testY, None, None, None)
        trainX, trainY, testX, testY = pre_mnist.get_preprocess_data()
        index = 10
        input_image = trainX[index]
        logger.debug("input shape: " + str(input_image.shape))
        trueLabel = np.argmax(trainY[index])
        logger.debug("True label: " + str(trueLabel))

        # gradient the neuron output corresponding to the true label w.r.t features
        input = tf.convert_to_tensor([input_image])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input)
            prediction_at_target_neuron = classifier(input)[0][trueLabel]
        gradient = tape.gradient(prediction_at_target_neuron, input)
        gradient = gradient.numpy()
        gradient = gradient.reshape(MNIST_IMG_ROWS * MNIST_IMG_COLS)

        # get the most important features
        n_important_features = 10
        for i in range(0, n_important_features):
            maxIndex = np.argmax(gradient)
            rowIndex = math.floor(maxIndex / MNIST_IMG_COLS)
            colIndex = maxIndex - rowIndex * MNIST_IMG_COLS
            logger.debug("max index = " + str(maxIndex) + "; max value = " + str(np.max(gradient))
                         + "; (" + str(rowIndex) + ", " + str(colIndex) + ")")
            input_image[rowIndex][colIndex][0] = 2
            gradient[maxIndex] = 0

        plt.imshow(input_image, cmap='gray')
        plt.title("Most important features are highlighted")
        plt.show()
