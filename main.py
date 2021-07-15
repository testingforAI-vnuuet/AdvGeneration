
import os
import sys
module_path = os.path.abspath(os.getcwd() + '/src')
if module_path not in sys.path:
    sys.path.append(module_path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from attacker.hpba import *
from utility.config import *
import tensorflow as tf

logger = MyLogger().getLog()

config_file = os.path.abspath('./config.ini')



if __name__ == '__main__':
    logger.debug('robustness START')
    logger.debug('reading configuration')
    analyze_config(config_file)
    logger.debug('reading configuration DONE')
    attacker = HPBA(origin_label=attack_config.original_class, trainX=attack_config.training_data,
                    trainY=attack_config.label_data, target_label=attack_config.target_class,
                    target_position=attack_config.target_position, classifier_name=attack_config.classifier_name,
                    weight=attack_config.weight, classifier=attack_config.classifier)
    attacker.export_result()
