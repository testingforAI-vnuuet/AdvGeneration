from __future__ import absolute_import

from numpy.core.multiarray import ndarray
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from attacker.constants import *
from attacker.losses import *
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *
import matplotlib.pyplot as plt

logger = MyLogger.getLog()


class MnistAutoEncoder:
    def __init__(self):
        pass

    def train(self,
              auto_encoder: keras.models.Model,
              attacked_classifier: keras.models.Model,
              loss,
              epochs: int,
              batch_size: int,
              epsilon: int,
              target_label: int,
              training_set: np.ndarray,
              output_model_path: str):
        # save the best model during training
        adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        earlyStopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(output_model_path, save_best_only=True, monitor='loss', mode='min')
        target_label_one_hot = keras.utils.to_categorical(target_label, MNIST_NUM_CLASSES, dtype='float32')
        auto_encoder.compile(optimizer=adam,
                             loss=loss(
                                 classifier=attacked_classifier,
                                 epsilon=epsilon,
                                 target_label=target_label_one_hot)
                             )
        auto_encoder.fit(x=training_set,
                         y=training_set,
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=[earlyStopping, mcp_save])
        return auto_encoder

    def get_architecture(self):
        input_img = keras.layers.Input(shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL))
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        decoded = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
        return keras.models.Model(input_img, decoded)

    def compute_balanced_point(self,
                               auto_encoder: Model,
                               attacked_classifier: Sequential,
                               loss,
                               train_data: ndarray,
                               target_label: int):
        target_label_one_hot = keras.utils.to_categorical(target_label, MNIST_NUM_CLASSES, dtype='float32')
        # compute the distance term
        auto_encoder.compile(loss=loss(
            classifier=attacked_classifier,
            epsilon=0,
            target_label=target_label_one_hot)
        )
        auto_encoder.fit(x=train_data,
                         y=train_data,
                         epochs=1,
                         batch_size=len(train_data))
        loss_distance = auto_encoder.history.history['loss'][0]

        # compute the probability term
        auto_encoder.compile(loss=loss(
            classifier=attacked_classifier,
            epsilon=1,
            target_label=target_label_one_hot)
        )
        auto_encoder.fit(x=train_data,
                         y=train_data,
                         epochs=1,
                         batch_size=len(train_data))
        loss_probability = auto_encoder.history.history['loss'][0]

        return loss_distance / (loss_distance + loss_probability)

    def plot(self, history: History, path: str):
        plt.plot(history.history['loss'])
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(path)
        plt.clf()


if __name__ == '__main__':
    START_SEED, END_SEED = 0, 1000
    TARGET = 7
    AE_LOSS = AE_LOSSES.cross_entropy_loss
    CNN_MODEL = keras.models.load_model(CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5')
    AE_MODEL = CLASSIFIER_PATH + '/xxxx.h5'
    FIG_PATH = CLASSIFIER_PATH + '/xxxx.png'

    # load dataset
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    pre_mnist = MnistPreprocessing(train_X, train_Y, test_X, test_Y, START_SEED, END_SEED, TARGET)
    train_X, train_Y, test_X, test_Y = pre_mnist.preprocess_data()
    countSamples(probability_vector=train_Y, n_class=MNIST_NUM_CLASSES)

    # train an autoencoder
    ae_trainer = MnistAutoEncoder()
    ae = ae_trainer.get_architecture()
    ae.summary()
    ae_trainer.train(
        auto_encoder=ae,
        attacked_classifier=CNN_MODEL,
        loss=AE_LOSS,
        epochs=2,
        batch_size=256,
        training_set=train_X,
        epsilon=0.01,
        output_model_path=AE_MODEL,
        output_loss_fig_path=FIG_PATH,
        target_label=TARGET)

    # compute the balance point
    balanced_point = ae_trainer.compute_balanced_point(auto_encoder=ae,
                                                       attacked_classifier=CNN_MODEL,
                                                       loss=AE_LOSS,
                                                       train_data=train_X,
                                                       target_label=TARGET)
    print(balanced_point)
