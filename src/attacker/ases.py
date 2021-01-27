from __future__ import absolute_import

import csv

from tensorflow.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from attacker.losses import *
from attacker.mnist_utils import get_advs, reject_outliers, preprocess_data
from constants import *
from data_preprocessing.mnist import MnistPreprocessing
from utility.statistics import *

logger = MyLogger.getLog()


class AEs:
    def __init__(self, train_images, train_labels, weights, input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL),
                 target=7):
        self.input_shape = input_shape
        self.ae_list = []
        self.trained_ae_list = []
        self.target_label = keras.utils.to_categorical(target, MNIST_NUM_CLASSES, dtype='float32')
        self.train_images = train_images
        self.train_labels = train_labels
        self.weights = weights

    def get_architecture_1(self):
        input_img = keras.Input(shape=self.input_shape)

        x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = keras.Model(input_img, decoded)

        return autoencoder

    def get_architecture_2(self, latentDim=16):
        input_img = keras.Input(shape=self.input_shape)
        x = keras.layers.Conv2D(32, (3, 3), strides=2, padding='same')(input_img)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv2D(64, (3, 3), strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.BatchNormalization()(x)

        volumeSize = K.int_shape(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(latentDim)(x)
        x = keras.layers.Dense(np.prod(volumeSize[1:]))(x)
        encoder = keras.layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        # decoder
        x = keras.layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(encoder)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Conv2DTranspose(MNIST_IMG_CHL, (3, 3), padding='same')(x)
        decoder = keras.layers.Activation('sigmoid')(x)

        autoencoder = keras.Model(input_img, decoder)

        return autoencoder

    def train(self, loss, epochs, batch_size, classifier):
        if len(self.ae_list) < 1:
            self.ae_list = [self.get_architecture_1()]

        for i, model in enumerate(self.ae_list):
            model_list_with_weights = []
            for weight in self.weights:
                output_model_path = 'saved_models/' + AUTOENCODER_FILE_FORMAT.format(index=i,
                                                                                     weight=str(weight).replace('.',
                                                                                                                ','))
                logger.debug("Training model: " + output_model_path)

                cloned_model = keras.models.clone_model(model)

                adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
                earlyStopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='min')
                mcp_save = ModelCheckpoint(output_model_path, save_best_only=True, monitor='loss', mode='min')

                cloned_model.compile(optimizer=adam,
                                     loss=loss(
                                         classifier=classifier,
                                         epsilon=weight,
                                         target_label=self.target_label))
                cloned_model.fit(self.train_images, self.train_images, epochs=epochs, batch_size=batch_size,
                                 callbacks=[earlyStopping, mcp_save])
                model_list_with_weights.append(cloned_model)
                logger.debug("Complete training model: " + output_model_path)
            self.trained_ae_list.append(model_list_with_weights)
        return

    def analyze(self, classifier, trainX, pretrained_models, name='aae'):

        for i, model_weights in enumerate(self.trained_ae_list):
            file_name = ANALYSED_FILE_NANE_FORMAT.format(name=name, num_model=str(i))
            result = []
            result.append(['weight', '#adv (success_rate)', 'L2'])

            for index, model_weight in enumerate(model_weights):
                row_epsilon = []
                generated_img_org = model_weight.predict(self.train_images)
                generated_img, gen_confident, trainX_new, confident_new = get_advs(classifier, self.train_images,
                                                                                   generated_img_org)
                l2 = np.array([np.linalg.norm(origin.flatten() - reconstruct.flatten()) for origin, reconstruct in
                               zip(trainX_new, generated_img)])
                l2txt = NODATA
                if l2.shape[0] > 0:
                    l2 = reject_outliers(l2)
                    l2txt = L2FORMAT.format(min=min(l2), max=max(l2), avg=np.average(l2))
                row_epsilon.append(self.weights[index])
                row_epsilon.append(len(generated_img))
                row_epsilon.append(l2txt)
                result.append(row_epsilon)

            for num_img in [10000, 20000, 40000]:
                result.append([num_img])
                for index, model_weight in enumerate(model_weights):
                    row_each_general = []
                    generated_img_org = model_weight.predict(trainX[:num_img])
                    generated_img, gen_confident, trainX_new, confident_new = get_advs(classifier, trainX[:num_img],
                                                                                       generated_img_org)
                    l2 = np.array([np.linalg.norm(origin.flatten() - reconstruct.flatten()) for origin, reconstruct in
                                   zip(trainX_new, generated_img)])

                    l2txt = NODATA

                    if l2.shape[0] > 0:
                        l2 = reject_outliers(l2)
                        l2txt = L2FORMAT.format(min=min(l2), max=max(l2), avg=np.average(l2))
                    row_each_general.append(self.weights[index])
                    row_each_general.append(len(generated_img) / float(num_img))
                    row_each_general.append(l2txt)
                    result.append(row_each_general)

            for i, pretrained_model in enumerate(pretrained_models):
                result.append(['pretrained_' + str(i)])
                for index, model_weight in enumerate(model_weights):
                    row_each_general = []
                    generated_img_org = model_weight.predict(self.train_images)
                    generated_img, gen_confident, trainX_new, confident_new = get_advs(pretrained_model, self.train_images,
                                                                                       generated_img_org)
                    l2 = np.array([np.linalg.norm(origin.flatten() - reconstruct.flatten()) for origin, reconstruct in
                                   zip(trainX_new, generated_img)])

                    l2txt = NODATA

                    if l2.shape[0] > 0:
                        l2 = reject_outliers(l2)
                        l2txt = L2FORMAT.format(min=min(l2), max=max(l2), avg=np.average(l2))
                    row_each_general.append(self.weights[index])
                    row_each_general.append(len(generated_img) / 1000)
                    row_each_general.append(l2txt)
                    result.append(row_each_general)


            with open(file_name, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(result)
        return


if __name__ == '__main__':
    START_SEED, END_SEED = 0, 1000
    TARGET = 7
    AE_LOSS = AE_LOSSES.cross_entropy_loss
    CNN_MODEL = keras.models.load_model(CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5')

    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()
    # pre_mnist = MnistPreprocessing(trainX, trainY, testX, testY, START_SEED, END_SEED, TARGET)
    # trainX, trainY, testX, testY = pre_mnist.preprocess_data()

    trainX, trainY = preprocess_data(trainX, trainY)
    testX, testY = preprocess_data(testX, testY)
    ae_weights = [0.001]

    AE = AEs(trainX[START_SEED:END_SEED], trainY[START_SEED:END_SEED], weights=ae_weights, target=TARGET)
    AE.train(loss=AE_LOSS, epochs=400, batch_size=256, classifier=CNN_MODEL)

    # Getting pretrained models
    pretrained_model_names = ['Alexnet', 'Lenet', 'vgg13', 'vgg16']
    pretrained_models = []
    for pretrained_model_name in pretrained_model_names:
        pretrained_models.append(
            keras.models.load_model(CLASSIFIER_PATH + '/pretrained_models/'+pretrained_model_name+'.h5'))
    AE.analyze(CNN_MODEL, trainX, pretrained_models)
    print('done')

