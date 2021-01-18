# @author: anh nguyen

# Implementation of MNIST model from the paper: https://arxiv.org/pdf/1511.04508.pdf (table I, II)

import os
import platform


import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential

if platform.platform().startswith('Darwin-19'):  # macosx
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MNIST:
    def __init__(self):
        return

    def get_architecture(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))
        return model

    def train(self, train_X, train_Y, path):
        model = self.get_architecture()
        opt = keras.optimizers.SGD(lr=0.1, momentum=0.5)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(train_X, train_Y, epochs=50, batch_size=128)

        model.save_weights(filepath=path)

    def load(self, test_X, test_Y, path):
        model = self.get_architecture()
        model.load_weights(path)
        Yhat = model.predict(test_X)
        print("Yhat.shape = ", Yhat.shape)  # (10000, 10)
        yhat = np.argmax(Yhat, axis=1)
        y = np.argmax(test_Y, axis=1)
        print("yhat.shape = ", yhat.shape)
        print("Accuracy on test set = ", np.average(yhat == y))

    def convert_to_onehotvector(self, test, nclass):
        """
        Convert 1-D array to 2-dimensional array using one-hot-vector transformation
        :param test: 1-D array
        :param nclass: a positive integer
        :return:
        """
        out = np.zeros((test.shape[0], nclass))
        print(out.shape)
        idx = 0
        for item in test:
            out[idx][item] = 1
            idx += 1
        return out


if __name__ == '__main__':
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()

    train_X = train_X.reshape(-1, 28, 28, 1)
    train_X = train_X / 255

    N_CLASSES = 10

    mymnist = MNIST()

    train_Y = mymnist.convert_to_onehotvector(train_Y, N_CLASSES)

    test_X = test_X.reshape(-1, 28, 28, 1)
    test_X = test_X / 255

    test_Y = mymnist.convert_to_onehotvector(test_Y, N_CLASSES)

    print("trainX shape = ", train_X.shape)
    print("train_Y.shape = ", train_Y.shape)

    model_path = "./pretrained_mnist_cnn1.h5"
    # mymnist.train(train_X, train_Y, model_path)
    mymnist.load(test_X, test_Y, model_path)
