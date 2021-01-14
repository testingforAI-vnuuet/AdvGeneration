from attacker.fgsm import *


class least_likely_class:
    def __init__(self, classifier, target, epsilon=0.001, num_iters=20):
        self.classifer = classifier
        self.target_label = keras.utils.to_categorical(target, num_classes=MNIST_NUM_CLASSES).reshape(
            (1, MNIST_NUM_CLASSES))

        self.epsilon = epsilon
        self.num_iters = num_iters

    def create_adv_single_image(self, image):
        for i in range(self.num_iters):
            grad = FGSM.create_adversarial_pattern(image, self.target_label, self.classifer, get_sign=False)
            image = image - self.epsilon * grad[0]
            image = np.clip(image, 0, 1)
        return image

    def create_adversaries(self, images):

        result = [self.create_adv_single_image(img) for img in images]
        return np.array(result)


if __name__ == '__main__':
    START_SEED = 0
    END_SEED = 1000
    TARGET = 7

    ATTACKED_CNN_MODEL = CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5'
    classifier = keras.models.load_model(ATTACKED_CNN_MODEL)

    (trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

    pre_mnist = mnist_preprocessing(trainX, trainY, testX, testY, START_SEED, END_SEED, TARGET)
    trainX, trainY, testX, testY = pre_mnist.get_preprocess_data()

    logger.debug('Creating adversarial examples: ')
    llc = least_likely_class(classifier=classifier, target=TARGET)
    result_imgs = llc.create_adversaries(trainX)
    result_origin_imgs, result_origin_confidients, result_gen_imgs, result_gen_confidents = filter_advs(classifier,
                                                                                                        trainX,
                                                                                                        result_imgs,
                                                                                                        TARGET)

    logger.debug('Least_likely_class done')
