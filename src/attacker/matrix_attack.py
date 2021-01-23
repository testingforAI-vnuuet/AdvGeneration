from __future__ import absolute_import

import json
import os
import pickle
import shutil

from tensorflow.keras.datasets import mnist

from attacker.au_adv_generator import generate_adv
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


def get_mnist():
    return get_root() + '/data/mnist'


def create_saved_folder():
    ROOT = get_root()
    logger.debug("ROOT = " + str(ROOT))
    if not os.path.exists(ROOT + '/data/'):
        os.mkdir(ROOT + '/data/')
    mnist_out = ROOT + '/data/mnist/'
    if not os.path.exists(mnist_out):
        os.mkdir(mnist_out)


def get_history_path():
    return get_mnist() + '/history.pkl'


def load_history_of_attack():
    path = get_history_path()
    if os.path.exists(path):
        history = json.load(open(path))
    else:
        history = dict()
    return history


if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 10000
    EPOCH = 1
    START_ATTACK_SEED = 10000
    END_ATTACK_SEED = 20000
    CNN_MODEL_PATH = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    CNN_MODEL = keras.models.load_model(CNN_MODEL_PATH)
    AE_LOSS = AE_LOSSES.cross_entropy_loss

    create_saved_folder()
    history = load_history_of_attack()

    for source_label in range(0, MNIST_NUM_CLASSES):
        # collect removed labels
        removed_labels = []
        for label in range(0, MNIST_NUM_CLASSES):
            if label != source_label:
                removed_labels.append(label)

        for target_label in range(0, MNIST_NUM_CLASSES):
            if source_label != target_label:
                name = str(source_label) + '_to_' + str(target_label)
                logger.debug("target label = " + str(target_label) + "; source label = " + str(source_label))
                if history.get(name) is not None:
                    logger.debug("Exist. Move the next attack!")
                else:
                    # create folder to save image
                    OUTPUT_AE_MODEL_PATH = get_mnist() + '/' + name + '.h5'
                    OUTPUT_LOSS_FIG_PATH = get_mnist() + '/' + name + '.png'

                    OUT_IMGAGES = get_mnist() + name
                    if os.path.exists(OUT_IMGAGES):
                        logger.debug("Clean " + OUT_IMGAGES)
                        shutil.rmtree(OUT_IMGAGES)
                    os.mkdir(OUT_IMGAGES)

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
                                                                       attacked_classifier=CNN_MODEL,
                                                                       loss=AE_LOSS,
                                                                       train_data=train_X,
                                                                       target_label=target_label)
                    logger.debug("balanced_point = " + str(balanced_point))
                    ae_trainer.train(
                        auto_encoder=ae,
                        attacked_classifier=CNN_MODEL,
                        loss=AE_LOSS,
                        epochs=EPOCH,
                        batch_size=512,
                        training_set=train_X,
                        epsilon=balanced_point,
                        output_model_path=OUTPUT_AE_MODEL_PATH,
                        output_loss_fig_path=OUTPUT_LOSS_FIG_PATH,
                        target_label=target_label)

                    # attack
                    # logger.debug("attack")
                    # (train_X2, train_Y2), (test_X2, test_Y2) = mnist.load_data()
                    # pre_mnist2 = MnistPreprocessing(train_X2, train_Y2, test_X2, test_Y2, START_ATTACK_SEED,
                    #                                 END_ATTACK_SEED,
                    #                                 removed_labels=removed_labels)
                    # train_X2, train_Y2, _, _ = pre_mnist2.preprocess_data()
                    # countSamples(probability_vector=train_Y2, n_class=MNIST_NUM_CLASSES)
                    #
                    # generate_adv(auto_encoder_path=OUTPUT_AE_MODEL_PATH,
                    #              loss=AE_LOSS,
                    #              cnn_model_path=CNN_MODEL_PATH,
                    #              train_X=train_X2,
                    #              train_Y=train_Y2,
                    #              should_clipping=True,
                    #              target=target_label,
                    #              out_image=OUT_IMGAGES)
                    history[name] = 1
                    json.dump(history, open(get_history_path(), 'w'))
