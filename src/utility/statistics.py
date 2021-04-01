import cv2
import numpy as np
from tensorflow import keras

from utility.mylogger import MyLogger

logger = MyLogger.getLog()


def countSamples(probability_vector, n_class):
    assert len(probability_vector.shape) == 2
    assert n_class >= 1

    table = np.zeros(n_class, dtype="int32")
    for item in probability_vector:
        table[np.argmax(item)] += 1

    # logger = MyLogger.getLog()
    logger.debug("Statistics:")
    logger.debug("Total: %s", probability_vector.shape[0])
    for item in range(0, n_class):
        logger.debug("\tLabel %s: %s samples (%s percentage)", item, table[item], table[item] / len(
            table))


def filter_by_label(label: int, data_set: np.ndarray, label_set: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    filtering data and label set by origin label

    :param label: origin label
    :type label: int
    :param data_set:
    :type data_set:
    :param label_set:
    :type label_set:
    :return: result_data, result_label
    :rtype:
    """
    logger.debug("Filtering for label: {label}".format(label=label))
    result_data = None
    result_label = None

    for data_i, label_i in zip(data_set, label_set):
        if np.argmax(label_i) == label:
            if result_data is None or result_label is None:
                result_data = np.array([data_i])
                result_label = np.array([label_i])
            else:
                result_data = np.append(result_data, [data_i], axis=0)
                result_label = np.append(result_label, [label_i], axis=0)
    logger.debug("Filtering for label end!")
    return result_data, result_label


def label_ranking(data_set: np.ndarray, classifier: keras.models.Model) -> np.ndarray:
    """
    ranking label by the chosen classifier

    :param data_set:
    :type data_set:
    :param classifier:
    :type classifier:
    :param label_position:
    :type label_position:
    :return:
    :rtype:
    """
    logger.debug("Ranking label")
    predictions = classifier.predict(data_set)
    ranked_array = np.argsort(predictions.sum(axis=0))
    logger.debug("Ranking label done!")
    return ranked_array


def filter_candidate_adv(origin_data: np.ndarray, candidate_adv: np.ndarray, target_label: int,
                         cnn_model: keras.models.Model):
    origin_confident = cnn_model.predict(origin_data)
    candidate_confident = cnn_model.predict(candidate_adv)

    result_adv_data = []
    result_adv_labels_vector = []

    result_origin_data = []
    result_origin_labels_vector = []

    for i in range(origin_data.shape[0]):
        if np.argmax(origin_confident[i]) != target_label and np.argmax(candidate_confident[i]) == target_label:
            result_adv_data.append(candidate_adv[i])
            result_adv_labels_vector.append(candidate_confident[i])

            result_origin_data.append(origin_data[i])
            result_origin_labels_vector.append(origin_confident[i])

    return list(
        map(np.array, [result_adv_data, result_adv_labels_vector, result_origin_data, result_origin_labels_vector]))


def compute_difference_two_set(set_1: np.ndarray, set_2: np.ndarray):
    result = np.sum([set_1_i - set_2_i for set_1_i, set_2_i in zip(set_1, set_2)], axis=0)
    return result / float(set_1.shape[0])


def get_border(images: np.ndarray) -> np.ndarray:
    border_results = []
    for image in images:
        border_img = (image * 255).astype(np.uint8)
        border_img = np.array(cv2.Canny(border_img, 100, 200)).reshape((28, 28, 1))
        border_results.append(border_img)
    return np.array(border_results, dtype=np.float32)/255.


def get_internal_images(images: np.ndarray, border_images=None) -> np.ndarray:
    internal_results = []
    if border_images is None:
        border_images = get_border(images)

    for border_image, image in zip(border_images, images):
        border_image_flat = border_image.flatten()
        image_flat = image.flatten()
        border_position = np.where(border_image_flat == 1.)
        internal_result = np.array(image_flat)
        internal_result[border_position] = 0
        internal_result = internal_result.reshape((28, 28, 1))

        internal_results.append(internal_result)

    return np.array(internal_results)


def ranking_sample(images,labels, origin_label, target_label, cnn_model: keras.models.Model):
    predictions = cnn_model.predict(images)
    priorities = np.array([abs(prediction[origin_label] - prediction[target_label]) for prediction in predictions])
    priority_indexes = np.argsort(priorities)

    return np.array(images[priority_indexes]), np.array(labels[priority_indexes])
