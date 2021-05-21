from attacker.mnist_utils import compute_l0_V2
from utility.feature_ranker import *


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
    avg_redundant_pixels = 0
    for generated_adv, origin_image in zip(generated_advs, origin_images):
        tmp_adv = np.array([generated_adv])
        for index in range(np.prod(generated_adv.shape)):
            row, col = int(index // 28), int(index % 28)
            tmp_value = tmp_adv[0][row, col]
            tmp_adv[0][row, col] = origin_image[row, col]
            predicted_label = np.argmax(classifier.predict(tmp_adv))
            if predicted_label != target_label:
                tmp_adv[0][row, col] = tmp_value
            else:
                avg_redundant_pixels += 1

        result.append(tmp_adv[0])

    return np.array(result), avg_redundant_pixels / float(generated_advs.shape[0])


def smooth_adv_border_V2(classifier, generated_advs, origin_images, border_indexs, target_label, step=1, K=784):
    sum_changed_pixels = 0
    restored_pixels_list = []

    for index, (generated_adv, origin_image, border_index) in enumerate(
            zip(generated_advs, origin_images, border_indexs)):
        tmp_adv = np.array([generated_adv])
        SX_flat = feature_ranker.jsma_ranking_border(origin_image, border_index, target_label, classifier)
        ranked_index = np.argsort(SX_flat)

        l0 = compute_l0_V2(generated_adv, origin_image)
        sum_changed_pixels += l0
        sum_restored_pixels_i = 0
        v_adv_j = []
        for i in range(1, K + 1, step):
            chosen_index = ranked_index[-1 * i * step]
            if SX_flat[chosen_index] == float('-inf'):
                break
            row, col = int(chosen_index // 28), int(chosen_index % 28)
            tmp_value = tmp_adv[0][row, col]
            tmp_adv[0][row, col] = origin_image[row, col]
            predicted_label = np.argmax(classifier.predict(tmp_adv))
            print(f'{predicted_label} {target_label}')
            if predicted_label != target_label:
                tmp_adv[0][row, col] = tmp_value
            else:
                sum_restored_pixels_i += 1
            v_adv_j.append(sum_restored_pixels_i/l0)

            # sum_restored_pixels_i_list.append(sum_restored_pixels_i)
        v_adv_j += [v_adv_j[-1]] * (K - len(v_adv_j))
        restored_pixels_list.append(v_adv_j)
    restored_pixels_list = np.array(restored_pixels_list)

    return np.average(restored_pixels_list, axis=0)

