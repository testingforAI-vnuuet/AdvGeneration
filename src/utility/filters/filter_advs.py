import numpy as np



def filter_advs(classifier, origin_images, generated_imgs, target):
    origin_confidents = classifier.predict(origin_images)
    gen_confidents = classifier.predict(generated_imgs)
    result_origin_imgs = []
    result_origin_confidents = []
    result_gen_imgs = []
    result_gen_confidents = []

    for i, origin_image in enumerate(origin_images):
        if np.argmax(origin_confidents[i]) != target and np.argmax(gen_confidents[i]) == target:
            result_origin_imgs.append(origin_image)
            result_origin_confidents.append(origin_confidents[i])
            result_gen_imgs.append(generated_imgs[i])
            result_gen_confidents.append(gen_confidents[i])
    return map(lambda data: np.array(data),
               [result_origin_imgs, result_origin_confidents, result_gen_imgs, result_gen_confidents])


def restore_redundant_mnist_pixels(classifier, generated_advs, origin_images, target_label):
    result = []

    for generated_adv, origin_image in zip(generated_advs, origin_images):
        tmp_adv = np.array([generated_adv])
        for index in generated_adv.reshape(np.prod(generated_adv.shape), ):
            row, col = index // 28, index % 28
            tmp_value = tmp_adv[0][row, col]
            tmp_adv[0][row, col] = origin_image[row, col]
            predicted_label = np.argmax(classifier.predict(tmp_adv))
            if predicted_label != target_label:
                tmp_adv[0][row, col] = tmp_value

        result.append(tmp_adv)

    return np.array(result)
