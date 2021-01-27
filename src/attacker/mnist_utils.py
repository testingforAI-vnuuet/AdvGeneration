import json
import os

import numpy as np
from matplotlib import pyplot as plt
from numpy.core.multiarray import ndarray
import tensorflow as tf
from constants import *

run_on_hpc = True


def get_root():
    if run_on_hpc:
        return "/home/anhnd/AdvGeneration"
    else:
        return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def get_autoencoder_output_path(name: str):
    return get_mnist_output() + '/model/' + name + '.h5'


def get_loss_output_path(name: str):
    return get_mnist_output() + '/model/' + name + '.png'


def get_mnist_output():
    """
    :return: `{root}/data/mnist`
    """
    mnist_out = get_root() + '/data/mnist/'
    if not os.path.exists(mnist_out):
        os.makedirs(mnist_out)
    return mnist_out


def get_history_path():
    return get_mnist_output() + '/history.pkl'


def get_image_output(name):
    path = get_mnist_output() + '/images/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_history_of_attack():
    path = get_history_path()
    if os.path.exists(path):
        his = json.load(open(path))
    else:
        his = dict()
    return his


def generate_identity_for_model(source_label: str,
                                target_label: str):
    return str(source_label) + '_to_' + str(target_label)


def plot(losses: ndarray, path: str):
    plt.plot(losses)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(path)
    plt.clf()


def get_advs(model, train, generated_img, target=7):
    # generated_img = autoencoder.predict(trainX_noise)
    confident = model.predict(train)
    gen_confident = model.predict(generated_img)
    generated_img_new = []
    confident_new = []
    gen_confident_new = []
    trainX_new = []
    for i in range(train.shape[0]):
        if np.argmax(confident[i]) != target and np.argmax(gen_confident[i]) == target:
            generated_img_new.append(generated_img[i])
            confident_new.append(confident[i])
            gen_confident_new.append(gen_confident[i])
            trainX_new.append(train[i])
    return generated_img_new, gen_confident_new, trainX_new, confident_new


def reject_outliers(data):
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return filtered


def preprocess_data(images, label, num_classes=MNIST_NUM_CLASSES, rols=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                    chl=MNIST_IMG_CHL):
    images = images.astype('float32') / 255.
    return images.reshape((len(images), rols, cols, chl)), \
           tf.keras.utils.to_categorical(label, num_classes)
