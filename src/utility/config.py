import configparser

import tensorflow as tf

from utility.constants import *
from utility.mylogger import *
from utility.utils import *

logger = MyLogger().getLog()


class attack_config:
    # classifier
    classifier_path = None
    classifier_name = None
    classifier = None

    # attack
    original_class = None
    target_class = None
    target_position = None
    recover_speed = None
    weight = None

    # data
    training_path = None
    training_data = None
    label_path = None
    label_data = None
    num_class = None
    input_shape = None
    input_size = None
    total_element_a_data = None


def analyze_config(config_path):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    #  classifier path
    classifier_path = os.path.abspath(config_parser['CLASSIFIER']['classifierPath'])
    if not check_path_exists(classifier_path):
        logger.error(f'not found classifier path: {classifier_path}')
    if not check_file_extension(classifier_path, MODEL_TF_EXTENSION):
        logger.error(
            f'file type does not match: {classifier_path}\n Please choose the file with extension: {MODEL_TF_EXTENSION}')
    attack_config.classifier_path = classifier_path
    attack_config.classifier = tf.keras.models.load_model(attack_config.classifier_path)
    attack_config.classifier_name = get_file_name(attack_config.classifier_path)

    # data path
    training_data_path = os.path.abspath(config_parser['DATA']['trainingDataPath'])
    if not check_path_exists(training_data_path):
        logger.error(f'not found training data path: {training_data_path}')
    if not check_file_extension(training_data_path, DATA_NP_EXTENSION):
        logger.error(
            f'file type does not match: {training_data_path}\n Please choose the file with extension: {DATA_NP_EXTENSION}')
    attack_config.training_path = training_data_path
    attack_config.training_data = np.load(attack_config.training_path)

    label_data_path = os.path.abspath(config_parser['DATA']['labelDataPath'])
    if not check_path_exists(label_data_path):
        logger.error(f'not found label data path: {label_data_path}')
    if not check_file_extension(label_data_path, DATA_NP_EXTENSION):
        logger.error(
            f'file type does not match: {label_data_path}\n Please choose file with extension: {DATA_NP_EXTENSION}')
    attack_config.label_data = label_data_path
    attack_config.label_data = np.load(label_data_path)

    if attack_config.training_data.shape[0] != attack_config.label_data.shape[0]:
        logger.error(f'training data and label data are not matched.')

    # analyze
    attack_config.input_size = attack_config.training_data.shape[0]
    data_example = attack_config.training_data[:1]
    attack_config.input_shape = data_example[0].shape
    attack_config.total_element_a_data = np.prod(attack_config.input_shape)
    attack_config.num_class = len(attack_config.classifier.predict(data_example)[0])

    # attack
    attack_config.original_class = int(config_parser['ATTACK']['originalClass'])
    if attack_config.original_class not in range(0, attack_config.num_class):
        logger.error(f'original class is not correct')

    target_class = int(config_parser['ATTACK']['targetClass'])
    target_position = int(config_parser['ATTACK']['targetPosition'])
    if target_class == attack_config.original_class:
        logger.error(f'target label should not be original label')
    if target_class not in range(0, attack_config.num_class):
        if target_position not in range(2, attack_config.num_class + 1):
            logger.error(f'target label or target position are not correct')
        else:
            attack_config.target_position = target_position
    else:
        attack_config.target_class = target_class
        attack_config.target_position = None

    recover_speed = float(config_parser['ATTACK']['recoverSpeed'])
    if recover_speed not in np.arange(0.1, 1, 0.1):
        logger.error(f'recover speed is not correct')
    attack_config.recover_speed = recover_speed

    weight = float(config_parser['ATTACK']['weight'])
    if weight not in np.arange(0.1, 1, 0.1):
        logger.error(f'weight if not correct')
    attack_config.weight = weight
