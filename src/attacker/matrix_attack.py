from __future__ import absolute_import

import json
import os
import shutil

from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.callbacks import History

from attacker.autoencoder import MnistAutoEncoder
from attacker.constants import *
from attacker.losses import *
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *

logger = MyLogger.getLog()

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


def load_history_of_attack():
    path = get_history_path()
    if os.path.exists(path):
        his = json.load(open(path))
    else:
        his = dict()
    return his


def generate_name_for_model(source_label: str,
                            target_label: str):
    return str(source_label) + '_to_' + str(target_label)


def attack(source_label: int,
           target_label: int,
           history: History,
           OUTPUT_AE_MODEL_PATH: str,
           OUTPUT_LOSS_FIG_PATH: str,
           ae_loss,
           cnn_model,
           name_model: str,
           epoch: int,
           batch: int):
    logger.debug("target label = " + str(target_label) + "; source label = " + str(source_label))
    if history.get(name_model) is not None:
        logger.debug("Exist. Move the next attack!")
    else:
        # process data
        logger.debug("preprocess mnist")
        (train_X, trainY), (test_X, test_Y) = mnist.load_data()
        pre_mnist = MnistPreprocessing(train_X, trainY, test_X, test_Y, START_SEED, END_SEED,
                                       removed_labels=removed_labels)
        train_X, trainY, test_X, test_Y = pre_mnist.preprocess_data()
        countSamples(probability_vector=trainY, n_class=MNIST_NUM_CLASSES)

        # train an autoencoder
        ae_trainer = MnistAutoEncoder()
        ae = ae_trainer.get_architecture()
        balanced_point = ae_trainer.compute_balanced_point(auto_encoder=ae,
                                                           attacked_classifier=cnn_model,
                                                           loss=ae_loss,
                                                           train_data=train_X,
                                                           target_label=target_label)
        logger.debug("balanced_point = " + str(balanced_point))
        ae_trainer.train(
            auto_encoder=ae,
            attacked_classifier=cnn_model,
            loss=ae_loss,
            epochs=epoch,
            batch_size=batch,
            training_set=train_X,
            epsilon=balanced_point / 4,
            output_model_path=OUTPUT_AE_MODEL_PATH,
            output_loss_fig_path=OUTPUT_LOSS_FIG_PATH,
            target_label=target_label)
        history[name] = 1
        json.dump(history, open(get_history_path(), 'w'))


if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 10000
    CNN_MODEL_PATH = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    CNN_MODEL = keras.models.load_model(CNN_MODEL_PATH)
    AE_LOSS = AE_LOSSES.cross_entropy_loss
    EPOCH = 400
    BATCH = 1024
    history = load_history_of_attack()

    for source_label in range(0, MNIST_NUM_CLASSES):
        # collect removed labels
        removed_labels = []
        for label in range(0, MNIST_NUM_CLASSES):
            if label != source_label:
                removed_labels.append(label)

        for target_label in range(0, MNIST_NUM_CLASSES):
            if source_label != target_label:
                name = generate_name_for_model(source_label, target_label)
                attack(source_label=source_label,
                       target_label=target_label,
                       history=history,
                       OUTPUT_AE_MODEL_PATH=get_autoencoder_output_path(name),
                       OUTPUT_LOSS_FIG_PATH=get_loss_output_path(name),
                       ae_loss=AE_LOSS,
                       cnn_model=CNN_MODEL,
                       name_model=name,
                       batch=BATCH,
                       epoch=EPOCH)
