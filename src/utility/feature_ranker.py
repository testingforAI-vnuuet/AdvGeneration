import enum
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import Sequential

from attacker.constants import MNIST_IMG_ROWS, MNIST_IMG_COLS, CLASSIFIER_PATH, MNIST_IMG_CHL
from data_preprocessing.mnist import MnistPreprocessing
from utility.mylogger import MyLogger

logger = MyLogger.getLog()


class RANKING_ALGORITHM(enum.Enum):
    ABS = 1
    COI = 2
    CO = 3


class feature_ranker:
    def __init__(self):
        return

    @staticmethod
    def compute_gradient_wrt_features(input: tf.Tensor,
                                      target_neuron: int,
                                      classifier: tf.keras.Sequential):
        """Compute gradient wrt features.

        Args:
            input: a tensor (shape = `[1, height, width, channel`])
            target_neuron: the index of the neuron on the output layer needed to be differentiated
            classifier: a sequential model
        Returns:
            gradient: ndarray (shape = `[height, width, channel`])
        """
        with tf.GradientTape() as tape:
            tape.watch(input)
            prediction_at_target_neuron = classifier(input)[0][target_neuron]
        gradient = tape.gradient(prediction_at_target_neuron, input)
        gradient = gradient.numpy()[0]
        return gradient

    @staticmethod
    def compute_gradient_batch(inputs: tf.Tensor, target_neuron: int, classifier: tf.keras.Sequential):
        '''

        :param inputs:
        :type inputs:
        :param target_neuron:
        :type target_neuron:
        :param classifier:
        :type classifier:
        :return:
        :rtype:
        '''
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions_at_target_neuron = classifier(inputs)[:, target_neuron]
        gradient = tape.gradient(predictions_at_target_neuron, inputs)
        return gradient.numpy()

    @staticmethod
    def random_ranking_batch(generated_advs, origin_images, target_label, classifier, diff_pixels, num_class):
        for index in range(len(diff_pixels)):
            random.shuffle(diff_pixels[index])
        return diff_pixels, None

    @staticmethod
    def coi_ranking_batch(generated_advs, origin_images, target_label, classifier, diff_pixels, num_class):
        dF_t = feature_ranker.compute_gradient_batch(inputs=tf.convert_to_tensor(generated_advs.reshape(-1, 28, 28, 1)),
                                                     classifier=classifier, target_neuron=target_label)
        dF_t = dF_t.reshape(-1, np.prod(generated_advs[0].shape))
        score_matrices = dF_t * generated_advs
        score_matrices = score_matrices

        ranking_results = []
        value_results = []
        for index in range(len(diff_pixels)):
            SX = score_matrices[index][diff_pixels[index]]
            a_argsort = np.argsort(SX)
            ranking_results.append(np.array(diff_pixels[index])[a_argsort])
            value_results.append(SX[a_argsort])
        return np.asarray(ranking_results), np.asarray(value_results)


        # return [random.shuffle(diff_i) for diff_i in diff_pixels], None

    @staticmethod
    def jsma_ranking_batch(generated_advs, origin_images, target_label, classifier, diff_pixels, num_class):
        dF_t = []
        dF_rest = []
        num_elements = np.prod(generated_advs[0].shape)
        for i in range(num_class):
            dF_i = feature_ranker.compute_gradient_batch(
                inputs=tf.convert_to_tensor(generated_advs.reshape((-1, 28, 28, 1))),
                classifier=classifier, target_neuron=target_label)
            if i != target_label:
                dF_rest.append(dF_i.reshape((-1, num_elements)))
            else:
                dF_t = dF_i.reshape((-1, num_elements))
        dF_rest = np.asarray(dF_rest)
        dF_rest = np.rollaxis(dF_rest, axis=1, start=0)
        dF_t = np.asarray(dF_t)
        advs_flatten = generated_advs.reshape((-1, num_elements))
        oris_flatten = origin_images.reshape((-1, num_elements))
        SXs = []

        for index in range(np.prod(generated_advs[0].shape)):
            dF_t_i = dF_t[:, index]
            sum_dF_rest_i = np.sum(dF_rest[:, :, index], axis=1)
            compare = advs_flatten[:, index] > oris_flatten[:, index]
            compare_opposite = ~ compare
            compare = np.array(compare, dtype=int)

            positive_rank = abs(dF_t_i) * abs(sum_dF_rest_i)
            negative_rank = np.zeros_like(positive_rank)

            init = np.array(positive_rank)
            init[compare is True and (dF_t_i < 0 or sum_dF_rest_i > 0)] = negative_rank[
                compare is True and dF_t_i < 0 and sum_dF_rest_i > 0]
            init[compare is False and (dF_t_i > 0 or sum_dF_rest_i < 0)] = negative_rank[
                compare is False and (dF_t_i > 0 or sum_dF_rest_i < 0)]
            SXs.append(init)

        SXs = np.asarray(SXs).T
        ranking_results = []
        value_results = []
        for index in range(len(diff_pixels)):
            SX = SXs[index][diff_pixels[index]]
            a_argsort = np.argsort(SX)
            ranking_results.append(np.array(diff_pixels[index])[a_argsort])
            value_results.append(SX[a_argsort])
        return np.asarray(ranking_results), np.asarray(value_results)

    @staticmethod
    def jsma_ka_ranking_batch(generated_advs, origin_images, target_label, classifier, diff_pixels, num_class):
        dF_t = []
        dF_rest = []
        num_elements = np.prod(generated_advs[0].shape)
        for i in range(num_class):
            dF_i = feature_ranker.compute_gradient_batch(
                inputs=tf.convert_to_tensor(generated_advs.reshape((-1, 28, 28, 1))),
                classifier=classifier, target_neuron=target_label)
            if i != target_label:
                dF_rest.append(dF_i.reshape((-1, num_elements)))
            else:
                dF_t = dF_i.reshape((-1, num_elements))
        dF_rest = np.asarray(dF_rest)
        dF_rest = np.rollaxis(dF_rest, axis=1, start=0)
        dF_t = np.asarray(dF_t)
        advs_flatten = generated_advs.reshape((-1, num_elements))
        oris_flatten = origin_images.reshape((-1, num_elements))
        SXs = []

        for index in range(num_elements):
            dF_t_i = dF_t[:, index]
            sum_dF_rest_i = np.sum(dF_rest[:, :, index], axis=1)
            compare = advs_flatten[:, index] > oris_flatten[:, index]
            compare_opposite = ~ compare
            compare = np.array(compare, dtype=int)
            compare_opposite = np.array(compare_opposite, dtype=int)

            positive_rank = abs(dF_t_i) * abs(sum_dF_rest_i)
            negative_rank = -1 * 1 / (positive_rank + 0.1)

            init = np.array(positive_rank)
            init[compare is True and (dF_t_i < 0 or sum_dF_rest_i > 0)] = negative_rank[
                compare is True and dF_t_i < 0 and sum_dF_rest_i > 0]
            init[compare is False and (dF_t_i > 0 or sum_dF_rest_i < 0)] = negative_rank[
                compare is False and (dF_t_i > 0 or sum_dF_rest_i < 0)]
            SXs.append(init)

        SXs = np.asarray(SXs).T
        ranking_results = []
        value_results = []
        for index in range(len(diff_pixels)):
            SX = SXs[index][diff_pixels[index]]
            a_argsort = np.argsort(SX)
            ranking_results.append(np.array(diff_pixels[index])[a_argsort])
            value_results.append(SX[a_argsort])
        return np.asarray(ranking_results), np.asarray(value_results)

    @staticmethod
    def find_important_features_of_samples(input_images: np.ndarray,
                                           n_rows: int, n_cols: int, n_channels: int, n_important_features: int,
                                           algorithm: enum.Enum,
                                           gradient_label: int,
                                           classifier: keras.Sequential):
        """Apply ranking algorithm to find the most important features.

        Args:
            input_images: an array of samples, `[size, height, width, channel`]
            n_rows: a positive number
            n_cols: a positive number
            n_channels: a positive number
            n_important_features: a positive number
            algorithm:
        Returns:
            positions: ndarray (shape=`[row, col, channel`])
        """
        final_important_features = np.ndarray(shape=(1, 3), dtype=int)
        for index in range(0, len(input_images)):
            input_image = input_images[index]
            important_features = feature_ranker.find_important_features_of_a_sample(
                n_rows=n_rows,
                n_cols=n_cols,
                n_channels=n_channels,
                input_image=input_image,
                n_important_features=n_important_features,
                algorithm=algorithm,
                gradient_label=gradient_label,
                classifier=classifier)
            final_important_features = np.concatenate(
                (final_important_features, important_features),
                axis=0)

        final_important_features = np.delete(
            arr=final_important_features,
            obj=0,
            axis=0)
        final_important_features = np.unique(final_important_features, axis=0)
        return final_important_features

    @staticmethod
    def highlight_important_features(important_features: np.ndarray, input_image: np.ndarray):
        """Highlight important features
            :param important_features: shape = '[ row, col, channel']. Each row stores the position of its feature on input image
            :param input_image: shape = `[ height, width, channel`]
        :return: None
        """
        input_image = input_image.copy()
        max = np.max(input_image)
        ROW_IDX = 0
        COL_IDX = 1
        CHANNEL_INDEX = 2
        for idx in range(0, important_features.shape[0]):
            row = important_features[idx, ROW_IDX]
            col = important_features[idx, COL_IDX]
            channel = important_features[idx, CHANNEL_INDEX]
            input_image[row, col, channel] = max + 2
        plt.imshow(input_image, cmap='gray')
        plt.title("Most important features are highlighted")
        plt.show()

    @staticmethod
    def jsma_ranking_border(generated_adv, origin_image, border_index, target_label, classifier,
                            num_expected_features=1,
                            num_classes=10):
        dF_t = None
        dF_rest = []
        for i in range(num_classes):
            dF_i = feature_ranker.compute_gradient_wrt_features(input=tf.convert_to_tensor([generated_adv]),
                                                                target_neuron=i, classifier=classifier)
            if i != target_label:
                dF_rest.append(dF_i)
            else:
                dF_t = dF_i

        SX = np.zeros_like(origin_image)
        for index in range(np.prod(origin_image.shape)):
            row, col = int(index // 28), int(index % 28)
            SX_i = None
            dF_t_i = dF_t[row, col]
            sum_dF_rest_i = sum([dF_rest_i[row, col] for dF_rest_i in dF_rest])
            if dF_t_i < 0 or sum_dF_rest_i > 0:
                SX_i = 0
            else:
                SX_i = dF_t_i * abs(sum_dF_rest_i)
            SX[row, col] = SX_i
        SX_flat = SX.flatten()
        SX_flat[border_index.flatten() == 0] = float('inf')
        return SX_flat

    @staticmethod
    def sequence_ranking(generated_adv, origin_image, border_index, target_label, classifier, num_expected_features=1,
                         num_classes=10):
        return np.array([*range(np.prod(generated_adv.shape, dtype=np.int64))])

    @staticmethod
    def jsma_ranking_borderV2(generated_adv, origin_image, border_index, target_label, classifier, diff_pixels,
                              num_expected_features=1,
                              num_classes=10):
        # compute gradient respect to generated_adv for each label
        dF_t = None  # gradient for target_label
        dF_rest = []  # array of gradient for the rest
        for i in range(num_classes):
            dF_i = feature_ranker.compute_gradient_wrt_features(
                input=tf.convert_to_tensor([generated_adv.reshape(28, 28, 1)]),
                target_neuron=i, classifier=classifier)
            if i != target_label:
                dF_rest.append(dF_i)
            else:
                dF_t = dF_i

        ori_2_dimension = origin_image.reshape(28, 28)
        adv_2_dimension = generated_adv.reshape(28, 28)
        # compute the importance of each pixel
        SX = np.zeros_like(origin_image)
        for index in range(np.prod(origin_image.shape)):
            row, col = int(index // 28), int(index % 28)
            dF_t_i = dF_t[row, col][0]
            sum_dF_rest_i = sum([abs(dF_rest_i[row, col][0]) for dF_rest_i in dF_rest])

            SX_i = 0
            if adv_2_dimension[row, col] > ori_2_dimension[row, col]:

                if dF_t_i < 0 or sum_dF_rest_i > 0:
                    SX_i = -1 * 1.0 / (abs(dF_t_i * sum_dF_rest_i) + 0.1)
                else:
                    SX_i = dF_t_i * abs(sum_dF_rest_i)
            else:
                if dF_t_i > 0 or sum_dF_rest_i < 0:
                    SX_i = -1 * 1.0 / (abs(dF_t_i * sum_dF_rest_i) + 0.1)
                else:
                    SX_i = abs(dF_t_i) * sum_dF_rest_i

            SX[row, col] = SX_i
            # print(f'dF_t_i={dF_t_i}')
            # print(f'sum_dF_rest_i={sum_dF_rest_i}')

        # get the rank of diff_pixels
        SX_flat = SX.flatten()
        a = SX_flat[diff_pixels]
        a_argsort = np.argsort(a)
        return np.array(diff_pixels)[a_argsort], a[a_argsort]

    @staticmethod
    def jsma_ranking_original(generated_adv, origin_image, border_index, target_label, classifier, diff_pixels,
                              num_expected_features=1,
                              num_classes=10):
        # compute gradient respect to generated_adv for each label
        dF_t = None  # gradient for target_label
        dF_rest = []  # array of gradient for the rest
        for i in range(num_classes):
            dF_i = feature_ranker.compute_gradient_wrt_features(
                input=tf.convert_to_tensor([generated_adv.reshape(28, 28, 1)]),
                target_neuron=i, classifier=classifier)
            if i != target_label:
                dF_rest.append(dF_i)
            else:
                dF_t = dF_i

        ori_2_dimension = origin_image.reshape(28, 28)
        adv_2_dimension = generated_adv.reshape(28, 28)
        # compute the importance of each pixel
        SX = np.zeros_like(origin_image)
        for index in range(np.prod(origin_image.shape)):
            row, col = int(index // 28), int(index % 28)
            dF_t_i = dF_t[row, col][0]
            sum_dF_rest_i = sum([abs(dF_rest_i[row, col][0]) for dF_rest_i in dF_rest])

            SX_i = 0
            if adv_2_dimension[row, col] > ori_2_dimension[row, col]:

                if dF_t_i < 0 or sum_dF_rest_i > 0:
                    SX_i = 0
                else:
                    SX_i = dF_t_i * abs(sum_dF_rest_i)
            else:
                if dF_t_i > 0 or sum_dF_rest_i < 0:
                    SX_i = 0
                else:
                    SX_i = abs(dF_t_i) * sum_dF_rest_i

            SX[row, col] = SX_i
            # print(f'dF_t_i={dF_t_i}')
            # print(f'sum_dF_rest_i={sum_dF_rest_i}')

        # get the rank of diff_pixels
        SX_flat = SX.flatten()
        a = SX_flat[diff_pixels]
        a_argsort = np.argsort(a)
        return np.array(diff_pixels)[a_argsort], a[a_argsort]

    @staticmethod
    def get_important_pixel_vetcan(image, classifier):

        important_pixels = []
        score = []
        changed_pixel_values = [2., 3.]
        tmp_image = np.array([image])
        predict_true = classifier.predict(tmp_image)[0]
        y_true = np.argmax(predict_true)
        confident_true = np.max(predict_true)

        for index in range(np.prod(image.shape)):
            row, col = int(index // 28), int(index % 28)
            tmp_pixel_value = tmp_image[0][row, col][0]
            for changed_pixel_value in changed_pixel_values:
                tmp_image[0][row, col] = changed_pixel_value
                predict = classifier.predict(tmp_image)[0]
                y_pred = np.argmax(predict)
                if y_pred != y_true:
                    print(f'pred: {y_pred}, true: {y_true}')
                    important_pixels.append(index)
                    score.append(abs(np.max(predict) - confident_true))
                    break
            tmp_image[0][row, col] = tmp_pixel_value

        return np.array(important_pixels), np.array(score)


if __name__ == '__main__':
    classifier = keras.models.load_model(CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5')
    LABEL = 4
    if isinstance(classifier, Sequential):
        # get a seed
        (trainX, trainY), (testX, testY) = mnist.load_data()
        pre_mnist = MnistPreprocessing(
            trainX=trainX,
            trainY=trainY,
            testX=testX,
            testY=testY,
            start=0,
            end=100,
            removed_labels=None)
        trainX, trainY, _, _ = pre_mnist.preprocess_data()

        # consider an input vector
        important_features = feature_ranker.find_important_features_of_a_sample(
            input_image=trainX[0],
            n_rows=MNIST_IMG_ROWS,
            n_cols=MNIST_IMG_COLS,
            n_channels=MNIST_IMG_CHL,
            n_important_features=50,
            algorithm=RANKING_ALGORITHM.ABS,
            gradient_label=3,
            classifier=classifier
        )
        logger.debug(important_features.shape)
        feature_ranker.highlight_important_features(
            important_features=important_features,
            input_image=trainX[0]
        )

        # consider input vectors
        important_features = feature_ranker.find_important_features_of_samples(
            input_images=trainX[0:100],
            n_rows=MNIST_IMG_ROWS,
            n_cols=MNIST_IMG_COLS,
            n_channels=MNIST_IMG_CHL,
            n_important_features=3,
            algorithm=RANKING_ALGORITHM.COI,
            gradient_label=1,
            classifier=classifier
        )
        logger.debug(important_features.shape)
        feature_ranker.highlight_important_features(
            important_features=important_features,
            input_image=trainX[1]
        )
