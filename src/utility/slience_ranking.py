"""
Created At: 10/06/2021 08:49

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utility.constants import *
from utility.helpers import normalize
from utility.mylogger import MyLogger

logger = MyLogger.getLog()


class slience_ranking:
    def __init__(self, classifier_name: str, classifier: tf.keras.models.Model, data_name=MNIST_DATA_NAME,
                 is_clip_above=True):
        """
        This is the implementation of paper: https://arxiv.org/pdf/1312.6034.pdf
        """

        self.classifier = classifier
        self.data_name = data_name
        self.result_image = None
        self.method_name = 'slience_ranking'
        self.is_clip_above = is_clip_above
        self.classifier_name = classifier_name

    def __compute_l2_in_tensor(self, image):
        image_flat = tf.reshape(tensor=image, shape=(-1))
        return sum(image_flat ** 2)

    def __get_loss(self, image, label, lamda):
        """

        :param image: image
        :type image: tf.Variable
        :param lamda: lamda
        :type lamda: float
        :return: term1 - lamda*term2, term1, term2
        :rtype: float, float, float
        """
        if len(image.shape) == 3:
            image = tf.reshape(tensor=image, shape=(-1, *image.shape))
        prediction = self.classifier(image)[0][label]
        l2 = self.__compute_l2_in_tensor(image)
        return prediction - lamda * l2, prediction, l2

    def compute_ranking_matrix(self, saved_image_path, label: int = 0, optimizer=optimizer_adam,
                               learning_rate: float = 0.1,
                               iteration: int = 50,
                               lamda: float = 0.5):

        """
        computing ranking matrix for each label
        :param optimizer: optimizer algorithm. Default: adam
        :type optimizer: str or tf.keras.optimizers
        :param learning_rate: initial learning rate for optimizer. Default: 0.1
        :type learning_rate: float
        :param iteration: number of iteration for update. Default: 100
        :type iteration: int
        :param lamda: weight for second terms. Default: 0.1
        :type lamda: float
        :return: ranking matrix
        :rtype: np.ndarray
        """
        logger.debug(f'[{self.method_name}]: ranking sample start!')
        init_image_shape = None  # pay attention to the exception
        if self.data_name == MNIST_DATA_NAME:
            init_image_shape = (MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)
        # implement CIFAR here
        else:
            logger.error(f'[{self.method_name}]: not found data name')
            raise NotImplemented('not found data name')
        init_image = np.zeros(shape=init_image_shape, dtype=float32_type)

        image = tf.Variable(init_image)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=10,
            decay_rate=0.9,
            staircase=True)

        opt = None
        if optimizer == optimizer_adam:
            opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        elif optimizer == optimizer_SGD:
            opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        else:
            logger.error(f'[{self.method_name}]: not found optimizer')
            raise NotImplemented('not found optimizer')
        image_per_iterations = []
        term_1_per_iterations = []
        term_2_per_iterations = []

        for i in range(iteration):
            image = tf.Variable(image)
            with tf.GradientTape() as tape:
                tape.watch(image)
                loss, term_1, term_2 = self.__get_loss(image=image, label=label, lamda=lamda)
                loss = -1 * loss
            print(f'iteration: {i}, loss: {loss}, lr: {opt._decayed_lr(tf.float32).numpy()}')

            grads = tape.gradient(loss, image)

            opt.apply_gradients(zip([grads], [image]))

            if self.is_clip_above:
                image = tf.clip_by_value(image, 0, float('inf'))
            else:
                image = tf.clip_by_value(image, float('-inf'), 0)

            if i % 10 == 0:
                image_per_iterations.append(image)
                term_1_per_iterations.append(term_1)
                term_2_per_iterations.append(term_2)

        self.result_image = image.numpy()
        image_results = [self.__post_processing(image_i) for image_i in image_per_iterations]
        shared_file_name = f'slience_matrix_{self.classifier_name}_label={label},optimizer={optimizer},lr={learning_rate},lamda={lamda}'
        self.__save_result_example(image_results, term_1_per_iterations, term_2_per_iterations, saved_image_path,
                                   shared_file_name)
        np.save(os.path.join(saved_image_path, shared_file_name + '.npy'), self.result_image)

    def __post_processing(self, image, need_to_normalize=False):
        image_np = image.numpy()
        image_np = image_np.reshape(-1)

        image_np[0] = 1

        if need_to_normalize:
            image_np = normalize(image_np)

        return image_np

    def __save_result_example(self, image_results, term_1_per_iterations, term_2_per_iterations, file_path,
                              shared_file_name):
        fig = plt.figure(figsize=(30, 4))
        n = len(image_results)
        for i in range(n):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(image_results[i].reshape((28, 28)), cmap='gray')
            title = f'iter={(i + 1) * 10}\n'
            title += f'term_1={round(term_1_per_iterations[i].numpy(), 0)}, term_2={round(term_2_per_iterations[i].numpy(), 0)}'
            ax.set_title(title)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig.suptitle(shared_file_name)
        plt.savefig(os.path.join(file_path, shared_file_name + '.png'))
        plt.close(fig)


if __name__ == '__main__':
    path_to_save = '../attacker/result/slience_map'
    classifier_name = 'Lenet_v2'
    classifier = tf.keras.models.load_model(f'../classifier/pretrained_models/{classifier_name}.h5')

    pre_softmax_classifier = tf.keras.models.Model(inputs=classifier.input,
                                                   outputs=classifier.get_layer('pre_softmax_layer').output)
    ranker = slience_ranking(classifier_name=classifier_name, classifier=pre_softmax_classifier,
                             data_name=MNIST_DATA_NAME,
                             is_clip_above=True)
    ranker.compute_ranking_matrix(saved_image_path=path_to_save, label=9)
