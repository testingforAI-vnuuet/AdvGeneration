from __future__ import absolute_import

import json
import os
import pickle
import shutil

from tensorflow.keras.datasets import mnist

from attacker.au_adv_generator import generate_adv
from attacker.autoencoder import Auto_encoder
from attacker.constants import *
from attacker.losses import *
from data_preprocessing.mnist import mnist_preprocessing
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

    START_ATTACK_SEED = 10000
    END_ATTACK_SEED = 20000

    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
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
                    AE_MODEL = CLASSIFIER_PATH + '/' + name + '.h5'
                    LOSS_FIG = CLASSIFIER_PATH + '/' + name + '.png'
                    OUT_IMGAGES = get_root() + '/data/mnist/' + name
                    if os.path.exists(OUT_IMGAGES):
                        logger.debug("Clean " + OUT_IMGAGES)
                        shutil.rmtree(OUT_IMGAGES)
                    os.mkdir(OUT_IMGAGES)

                    # process data
                    logger.debug("preprocess mnist")
                    (trainX, trainY), (testX, testY) = mnist.load_data()
                    pre_mnist = mnist_preprocessing(trainX, trainY, testX, testY, START_SEED, END_SEED,
                                                    removed_labels=removed_labels)
                    trainX, trainY, testX, testY = pre_mnist.get_preprocess_data()
                    countSamples(probabilityVector=trainY, nClasses=MNIST_NUM_CLASSES)

                    # train an autoencoder
                    auto_encoder = Auto_encoder(train_data=trainX, target=target_label, nClasses=MNIST_NUM_CLASSES)
                    auto_encoder.set_output_file(AE_MODEL)
                    auto_encoder.set_loss_fig(LOSS_FIG)
                    classifier = keras.models.load_model(ATTACKED_CNN_MODEL)
                    auto_encoder.compile(classifier=classifier, loss=AE_LOSS)
                    auto_encoder.fit(epochs=1, batch_size=512)

                    # attack
                    logger.debug("attack")
                    (trainX2, trainY2), (testX2, testY2) = mnist.load_data()
                    pre_mnist2 = mnist_preprocessing(trainX2, trainY2, testX2, testY2, START_ATTACK_SEED,
                                                     END_ATTACK_SEED,
                                                     removed_labels=removed_labels)
                    trainX2, trainY2, _, _ = pre_mnist2.get_preprocess_data()
                    countSamples(probabilityVector=trainY2, nClasses=MNIST_NUM_CLASSES)

                    generate_adv(autoencoder=AE_MODEL,
                                 loss=AE_LOSSES.cross_entropy_loss,
                                 cnn_model=ATTACKED_CNN_MODEL,
                                 trainX=trainX2,
                                 trainY=trainY2,
                                 should_clipping=True,
                                 target=target_label,
                                 out_image=OUT_IMGAGES)
                    history[name] = 1
                    json.dump(history, open(get_history_path(), 'w'))
