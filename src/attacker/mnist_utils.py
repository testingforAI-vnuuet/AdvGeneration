import json
import os
import os
from matplotlib import pyplot as plt
from numpy.core.multiarray import ndarray

from numpy.core.multiarray import ndarray

run_on_hpc = False

history = {}


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
