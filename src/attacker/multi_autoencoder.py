from __future__ import absolute_import

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, History

from attacker.constants import *
from attacker.losses import *
from utility.statistics import *

matplotlib.use('TkAgg')

logger = MyLogger.getLog()


class MultiMnistAutoEncoder:
    def __init__(self):
        pass

    def train(self,
              auto_encoder: keras.models.Model,
              attacked_classifier: keras.models.Model,
              loss,
              epochs: int,
              batch_size: int,
              epsilon: float,
              true_label: int,
              training_set: np.ndarray,
              output_model_path: str,
              is_fit=True):
        # save the best model during training
        adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        earlyStopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(output_model_path, save_best_only=True, monitor='loss', mode='min')

        losses = []
        for idx in range(MNIST_NUM_CLASSES):
            if idx != true_label:
                target_label_one_hot = keras.utils.to_categorical(idx, MNIST_NUM_CLASSES, dtype='float32')
                losses.append(
                    loss(
                        classifier=attacked_classifier,
                        epsilon=epsilon,
                        target_label=target_label_one_hot)
                )

        auto_encoder.compile(optimizer=adam, loss=losses)
        if is_fit:
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

        outputs = []
        for idx in range(MNIST_NUM_CLASSES - 1):
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
            x = keras.layers.UpSampling2D((2, 2))(x)
            x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
            decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
            outputs.append(decoded)

        return keras.models.Model(input_img, outputs)

    def plot(self, history: History, path: str):
        plt.plot(history.history['loss'])
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(path)
        plt.clf()


def get_X_attack(X_train, y_train, wrongseeds, ORI_LABEL, N_ATTACKING_SAMPLES, start=None, end=None):
    count = 0
    X_attack = []
    if end is None:
        end = len(X_train)
    if start is None:
        start = 0

    selected_seed = []
    for idx in range(start, end):
        if idx not in wrongseeds and y_train[idx] == ORI_LABEL:
            X_attack.append(X_train[idx])
            selected_seed.append(idx)
            count += 1
            if count == N_ATTACKING_SAMPLES:
                break
    X_attack = np.asarray(X_attack)
    selected_seed = np.asarray(selected_seed)
    # X_attack = X_attack[:N_ATTACKING_SAMPLES]
    # print(f'The shape of X_attack = {X_attack.shape}')
    return X_attack, selected_seed


def show_ten_images(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                    x1_title="", x2_title="", x3_title="", x4_title="", x5_title="", x6_title="", x7_title="",
                    x8_title="", x9_title="", x10_title=""):
    fig = plt.figure()
    fig.add_subplot(2, 5, 1).title.set_text(x1_title)
    plt.imshow(x1.reshape(28, 28), cmap="gray")

    fig.add_subplot(2, 5, 2).title.set_text(x2_title)
    plt.imshow(x2.reshape(28, 28), cmap="gray")

    fig.add_subplot(2, 5, 3).title.set_text(x3_title)
    plt.imshow(x3.reshape(28, 28), cmap="gray")

    fig.add_subplot(2, 5, 4).title.set_text(x4_title)
    plt.imshow(x4.reshape(28, 28), cmap="gray")

    fig.add_subplot(2, 5, 5).title.set_text(x5_title)
    plt.imshow(x5.reshape(28, 28), cmap="gray")

    fig.add_subplot(2, 5, 6).title.set_text(x6_title)
    plt.imshow(x6.reshape(28, 28), cmap="gray")

    fig.add_subplot(2, 5, 7).title.set_text(x7_title)
    plt.imshow(x7.reshape(28, 28), cmap="gray")

    fig.add_subplot(2, 5, 8).title.set_text(x8_title)
    plt.imshow(x8.reshape(28, 28), cmap="gray")

    fig.add_subplot(2, 5, 9).title.set_text(x9_title)
    plt.imshow(x9.reshape(28, 28), cmap="gray")

    fig.add_subplot(2, 5, 10).title.set_text(x10_title)
    plt.imshow(x10.reshape(28, 28), cmap="gray")

    plt.show()


if __name__ == '__main__':
    N_ATTACKING_SAMPLES = None
    ORI_LABEL = 2
    AE_LOSS = AE_LOSSES.cross_entropy_loss
    CNN_MODEL = keras.models.load_model(CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5')
    AE_MODEL = CLASSIFIER_PATH + '/xxxx.h5'
    FIG_PATH = CLASSIFIER_PATH + '/xxxx.png'

    (X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    X_train = X_train / 255

    X_attack, selected_seeds = get_X_attack(X_train, y_train, [], ORI_LABEL, N_ATTACKING_SAMPLES)
    X_attack = X_attack.reshape(-1, 28, 28, 1)

    TRAIN = True
    if TRAIN:
        # train an autoencoder
        ae_trainer = MultiMnistAutoEncoder()
        ae = ae_trainer.get_architecture()
        ae.summary()
        # plot_model(ae, to_file='/Users/ducanhnguyen/Documents/python/pycharm/AdvGeneration/data/model_plot.png', show_shapes=True, show_layer_names=True)

        ae_trainer.train(
            auto_encoder=ae,
            attacked_classifier=CNN_MODEL,
            loss=AE_LOSS,
            epochs=500,
            batch_size=512,
            training_set=X_attack,
            epsilon=0.01,
            output_model_path=AE_MODEL,
            true_label=ORI_LABEL)
    else:
        ae = keras.models.load_model(filepath=AE_MODEL, compile=False)
        for idx in range(0, 10):
            an_input = X_attack[idx:idx + 1]
            outputs = ae.predict(an_input)
            outputs = np.asarray(outputs)
            outputs = outputs.reshape(-1, 28, 28, 1)
            Y_preds = CNN_MODEL.predict(outputs)
            y_pred = np.argmax(Y_preds, axis=1)
            show_ten_images(
                an_input, outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
                outputs[7], outputs[8],
                x1_title="origin",
                x2_title=f"pred: {y_pred[0]}\ntarget: 0",
                x3_title=f"pred: {y_pred[1]}\ntarget: 1",
                x4_title=f"pred: {y_pred[2]}\ntarget: 3",
                x5_title=f"pred: {y_pred[3]}\ntarget: 4",
                x6_title=f"pred: {y_pred[4]}\ntarget: 5",
                x7_title=f"pred: {y_pred[5]}\ntarget: 6",
                x8_title=f"pred: {y_pred[6]}\ntarget: 7",
                x9_title=f"pred: {y_pred[7]}\ntarget: 8",
                x10_title=f"pred: {y_pred[8]}\ntarget: 9"
            )
    # print(outputs)