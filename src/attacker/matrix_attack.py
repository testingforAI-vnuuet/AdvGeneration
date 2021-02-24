from __future__ import absolute_import

import json

from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.callbacks import History

from attacker.au_adv_generator import generate_adv
from attacker.autoencoder import MnistAutoEncoder
from attacker.constants import *
from attacker.losses import *
from attacker.mnist_utils import get_history_path, load_history_of_attack, generate_identity_for_model, \
    get_autoencoder_output_path, get_loss_output_path, get_image_output, plot
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *

logger = MyLogger.getLog()


def attack(source_label: int,
           target_label: int,
           history: History,
           output_ae_model_path: str,
           output_loss_fig_path: str,
           ae_loss,
           cnn_model,
           name_model: str,
           epoch: int,
           batch: int):
    logger.debug("target label = " + str(target_label) + "; source label = " + str(source_label))
    if history.get(name_model) is not None:
        logger.debug("Exist. Move the next attack!")
        return False
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

        N_ITER = 4
        losses = []
        for i in range(N_ITER):
            logger.debug("Iterate " + str(i))
            balanced_point = ae_trainer.compute_balanced_point(auto_encoder=ae,
                                                               attacked_classifier=cnn_model,
                                                               loss=ae_loss,
                                                               train_data=train_X,
                                                               target_label=target_label)
            logger.debug("balanced_point = " + str(balanced_point))
            ae = ae_trainer.train(
                auto_encoder=ae,
                attacked_classifier=cnn_model,
                loss=ae_loss,
                epochs=int(epoch / N_ITER),
                batch_size=batch,
                training_set=train_X,
                epsilon=balanced_point,
                output_model_path=output_ae_model_path,
                target_label=target_label)
            for item in ae.history.history['loss']:
                losses.append(item)

        plot(losses, output_loss_fig_path)
        history[identity] = 1
        json.dump(history, open(get_history_path(), 'w'))
        return True


if __name__ == '__main__':
    START_ATTACK_SEED, END_ATTACK_SEED = 10000, 12000

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

        #
        for target_label in range(0, MNIST_NUM_CLASSES):
            if source_label != target_label:
                identity = generate_identity_for_model(source_label, target_label)
                be_attacked = attack(source_label=source_label,
                                     target_label=target_label,
                                     history=history,
                                     output_ae_model_path=get_autoencoder_output_path(identity),
                                     output_loss_fig_path=get_loss_output_path(identity),
                                     ae_loss=AE_LOSS,
                                     cnn_model=CNN_MODEL,
                                     name_model=identity,
                                     batch=BATCH,
                                     epoch=EPOCH)

                #
                if (be_attacked):
                    (train_X2, train_Y2), (test_X2, test_Y2) = mnist.load_data()
                    pre_mnist2 = MnistPreprocessing(train_X2, train_Y2, test_X2, test_Y2, START_ATTACK_SEED,
                                                    END_ATTACK_SEED,
                                                    removed_labels=removed_labels)
                    train_X2, train_Y2, _, _ = pre_mnist2.preprocess_data()
                    countSamples(probability_vector=train_Y2, n_class=MNIST_NUM_CLASSES)

                    generate_adv(auto_encoder_path=get_autoencoder_output_path(identity),
                                 loss=AE_LOSS,
                                 cnn_model_path=CNN_MODEL_PATH,
                                 train_X=train_X2,
                                 train_Y=train_Y2,
                                 should_clipping=True,
                                 target=target_label,
                                 out_image=get_image_output(identity))
