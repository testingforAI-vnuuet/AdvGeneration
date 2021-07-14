import os

MNIST_IMG_ROWS = 28
MNIST_IMG_COLS = 28
MNIST_IMG_CHL = 1
MNIST_NUM_CLASSES = 10

CLASSIFIER_PATH = '../classifier'

PRETRAIN_CLASSIFIER_PATH = CLASSIFIER_PATH + '/pretrained_models'
MNIST_CNN_MODEL_WITH_PRE_SOFTMAX_FILE = '/MNIST_MODEl_WITH_PRE_SOFTMAX.h5'

RESULT_FOLDER_PATH = os.path.abspath('../../results')
SAVED_ATTACKER_PATH = './saved_models'
SAVED_IMAGE_SAMPLE_PATH = './saved_images'
SAVED_NPY_PATH = './saved_npy'

PRIMARY_BORDER_METHOD_NAME = 'ae_border'
BORDER_METHOD_NAME = 'ae_border'
SLIENCE_MAP_METHOD_NAME = 'ae_slience_map'
AE4DNN_METHOD_NAME = 'ae4dnn'
AAE_METHOD_NAME = 'aae'
FGSM_METHOD_NAME = 'fgsm'
LBFGS_METHOD_NAME = 'lbfgs'
HPBA_METHOD_NAME = 'hpba'




ALL_PATTERN = 'all_pixel_pattern'
BORDER_PATTERN = 'border_pattern'
SALIENCE_PATTERN = 'salience_pattern'


TEXT_AUTOENCODER = 'autoencoder'
TEXT_IMAGE= 'image'
TEXT_RESULT_SUMMARY = 'result_summary'
TEXT_DATA = 'data'
