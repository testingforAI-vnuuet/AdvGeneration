import numpy as np


def filter_advs(classifier, origin_images, generated_imgs, target):
    origin_confidents = classifier.predict(origin_images)
    gen_confidents = classifier.predict(generated_imgs)
    result_origin_imgs = []
    result_origin_confidients = []
    result_gen_imgs = []
    result_gen_confidents = []

    for i, origin_image in enumerate(origin_images):
        if np.argmax(origin_confidents[i]) != target and np.argmax(gen_confidents[i]) == target:
            result_origin_imgs.append(origin_image)
            result_origin_confidients.append(origin_confidents[i])
            result_gen_imgs.append(generated_imgs[i])
            result_gen_confidents.append(gen_confidents[i])
    return map(lambda data: np.array(data), [result_origin_imgs, result_origin_confidients, result_gen_imgs, result_gen_confidents])

