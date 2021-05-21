import json
import os

import numpy as np
from matplotlib import pyplot as plt
from numpy.core.multiarray import ndarray

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


def reject_outliers(data):
    if data.shape[0] == 0:
        return np.array([])
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return np.array(filtered)


def L0(gen, test):
    threshold = 10e-4
    gen_new = np.array(gen)
    test_new = np.array(test)

    return sum(0 if abs(g - t) < threshold else 1 for g, t in zip(gen_new.flatten(), test_new.flatten()))

def compute_l0_V2(adv: np.ndarray,
               ori: np.ndarray,
               normalized=False):  # 1d array, value in range of [0 .. 1]
    if not normalized:
        adv = np.round(adv * 255)
        ori = np.round(ori * 255)
    adv = adv.reshape(-1)
    ori = ori.reshape(-1)
    l0_dist = 0
    for idx in range(len(adv)):
        if adv[idx] != ori[idx]:
            l0_dist += 1
    return l0_dist


def L2(gen, test):
    gen = gen.flatten()
    test = test.flatten()
    dist = (gen - test) ** 2
    dist = np.sum(dist)
    return np.sqrt(dist)


def plot(losses: ndarray, path: str):
    plt.plot(losses)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig(path)
    plt.clf()
