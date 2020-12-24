from .constants import *
from tensorflow import keras
from .losses import *


class Auto_encoder:
    def __init__(self, target=7):
        self.auto_encoder = None
        self.target = target
        self.target_label_onehot = keras.utils.to_categorical(target, MNIST_NUM_CLASSES, dtype='float32')
        pass

    def define_architecture(self, input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)):
        input_img = keras.layers.Input(shape=input_shape)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        self.auto_encoder = keras.models.Model(input_img, decoded)

    def compile(self, classifier, loss):
        if not isinstance(self.auto_encoder, keras.models.Model):
            raise AttributeError("Define Autoencoder architecture first")

        adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.auto_encoder.compile(optimizer=adam, loss=loss(classifier, self.target_label_onehot),
                                  metrics=['accuracy'])

    def fit(self, seed_images):
        # compile first
        self.auto_encoder.fit(seed_images, seed_images, epochs=20, batch_size=100)



