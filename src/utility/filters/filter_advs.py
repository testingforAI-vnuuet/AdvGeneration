import os

from attacker.mnist_utils import compute_l0_V2, compute_l2_V2
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
        SX_flat = feature_ranker.sequence_ranking(generated_adv, origin_image, border_index, target_label, classifier)
        ranked_index = np.argsort(SX_flat)[::-1]

        l0 = compute_l0_V2(generated_adv, origin_image)
        sum_changed_pixels += l0
        sum_restored_pixels_i = 0
        v_adv_j = []
        for i in range(1, K + 1, step):
            chosen_index = ranked_index[-1 * i * step]
            if SX_flat[chosen_index] == float('inf'):
                break
            row, col = int(chosen_index // 28), int(chosen_index % 28)
            if generated_adv[row, col] == origin_image[row, col]:
                continue
            tmp_value = tmp_adv[0][row, col]
            tmp_adv[0][row, col] = origin_image[row, col]
            predicted_label = np.argmax(classifier.predict(tmp_adv))
            # print(f'{predicted_label} {target_label}')
            if predicted_label != target_label:
                tmp_adv[0][row, col] = tmp_value
            else:
                sum_restored_pixels_i += 1
            v_adv_j.append(sum_restored_pixels_i / l0)

            # sum_restored_pixels_i_list.append(sum_restored_pixels_i)
        v_adv_j += [v_adv_j[-1]] * (K - len(v_adv_j))
        restored_pixels_list.append(v_adv_j)
    restored_pixels_list = np.array(restored_pixels_list)

    return np.average(restored_pixels_list, axis=0)


def rank_pixel_S2(diff_pixel):  # top-left to bottom-right
    return diff_pixel


def smooth_vet_can_step(ori, adv, dnn, target_label, step, strategy):
    n_restored_pixels = 0
    restored_pixel_by_prediction = []

    # normalize
    ori_0_255 = ori.reshape(-1)
    smooth_adv_0_255 = np.array(adv).reshape(-1)
    original_adv_0_255 = np.array(adv).reshape(-1)
    if np.min(ori) >= 0 and np.max(ori) <= 1:
        ori_0_255 = np.round(ori_0_255 * 255)
        smooth_adv_0_255 = np.round(smooth_adv_0_255 * 255)
        original_adv_0_255 = np.round(original_adv_0_255 * 255)

    L0_before = compute_l0_V2(smooth_adv_0_255, ori_0_255, normalized=True)
    print(f"L0_before = {L0_before}")

    # get different pixels
    diff_pixel_arr = []
    for diff_pixel_idx in range(len(ori_0_255)):
        if ori_0_255[diff_pixel_idx] != smooth_adv_0_255[diff_pixel_idx]:
            diff_pixel_arr.append(diff_pixel_idx)

    # diff_pixel_arr = rank_pixel_S2(diff_pixel_arr)
    diff_pixel_arr = feature_ranker.jsma_ranking_borderV2(adv, ori, None, target_label, dnn, diff_pixel_arr)

    diff_pixel_arr = np.asarray(diff_pixel_arr)

    #
    count = 0
    old_indexes = []
    old_values = []
    for diff_pixel_idx in diff_pixel_arr:
        if ori_0_255[diff_pixel_idx] != smooth_adv_0_255[diff_pixel_idx]:
            count += 1
            old_indexes.append(diff_pixel_idx)
            old_values.append(smooth_adv_0_255[diff_pixel_idx])
            smooth_adv_0_255[diff_pixel_idx] = ori_0_255[diff_pixel_idx]  # try to restore

        if count == step \
                or diff_pixel_idx == diff_pixel_arr[-1]:
            Y_pred = dnn.predict((smooth_adv_0_255 / 255).reshape(-1, 28, 28, 1))
            pred = np.argmax(Y_pred, axis=1)[0]

            if pred != target_label:
                # revert the changes
                for jdx, value in zip(old_indexes, old_values):
                    smooth_adv_0_255[jdx] = value
            else:
                n_restored_pixels += count

            old_indexes = []
            old_values = []
            count = 0
            restored_pixel_by_prediction.append(n_restored_pixels)

    L0_after = compute_l0_V2(ori_0_255, smooth_adv_0_255, normalized=True)
    print(f"L0_after = {L0_after}")

    L2_after = compute_l0_V2(ori_0_255, smooth_adv_0_255)
    L2_before = compute_l2_V2(ori_0_255, original_adv_0_255)

    # highlight = utilities.highlight_diff(original_adv_0_255, smooth_adv_0_255)

    return smooth_adv_0_255, L0_after, L0_before, L2_after, L2_before, np.asarray(
        restored_pixel_by_prediction)


def smooth_vet_can_stepV2(ori, adv, dnn, target_label, step, strategy=None):
    n_restored_pixels = 0
    restored_pixel_by_prediction = []

    # normalize
    ori_0_255 = ori.reshape(-1)
    smooth_adv_0_255 = np.array(adv).reshape(-1)
    original_adv_0_255 = np.array(adv).reshape(-1)
    if np.min(ori) >= 0 and np.max(ori) <= 1:
        ori_0_255 = np.round(ori_0_255 * 255)
        smooth_adv_0_255 = np.round(smooth_adv_0_255 * 255)
        original_adv_0_255 = np.round(original_adv_0_255 * 255)

    L0_before = compute_l0_V2(smooth_adv_0_255, ori_0_255, normalized=True)
    print(f"L0_before = {L0_before}")

    # get different pixels
    diff_pixel_arr = []
    for diff_pixel_idx in range(len(ori_0_255)):
        if ori_0_255[diff_pixel_idx] != smooth_adv_0_255[diff_pixel_idx]:
            diff_pixel_arr.append(diff_pixel_idx)

    # diff_pixel_arr = rank_pixel_S2(diff_pixel_arr)
    diff_pixel_arr, diff_value_arr = feature_ranker.jsma_ranking_borderV2(adv, ori, None, target_label, dnn,
                                                                          diff_pixel_arr)

    curr_diff_pixel_arr = np.array(diff_pixel_arr)
    curr_diff_value_arr = np.array(diff_value_arr)
    old_pixels = np.array([], dtype=np.int32)
    old_values = np.array([])
    count_step = 0
    count_changes = 0
    while curr_diff_pixel_arr is not None:
        count_step += 1
        curr_pixels, curr_diff_pixel_arr, curr_diff_value_arr = get_all_same_element_by_index(curr_diff_pixel_arr,
                                                                                              curr_diff_value_arr)
        old_pixels = np.concatenate((old_pixels, curr_pixels))
        old_values = np.concatenate((old_values, smooth_adv_0_255[curr_pixels]))
        count_changes += 0 if curr_pixels is None else len(curr_pixels)

        smooth_adv_0_255[curr_pixels] = ori_0_255[curr_pixels]
        # L0_after = compute_l0_V2(smooth_adv_0_255, ori_0_255, normalized=True)
        # print(f'{L0_after} is L0 after')
        # print(len(curr_pixels))

        if count_step == step or curr_diff_pixel_arr is None:
            Y_pred = dnn.predict((smooth_adv_0_255 / 255.).reshape(-1, 28, 28, 1))
            pred = np.argmax(Y_pred, axis=1)[0]
            if pred != target_label:
                smooth_adv_0_255[old_pixels] = old_values
            else:
                n_restored_pixels += count_changes
            old_pixels = np.array([], dtype=np.int32)
            old_values = np.array([])
            count_step = 0
            count_changes = 0
            restored_pixel_by_prediction.append(n_restored_pixels)

    L0_after = compute_l0_V2(ori_0_255, smooth_adv_0_255, normalized=True)
    print(f"L0_after = {L0_after}")

    L2_after = compute_l0_V2(ori_0_255, smooth_adv_0_255)
    L2_before = compute_l2_V2(ori_0_255, original_adv_0_255)

    # highlight = utilities.highlight_diff(original_adv_0_255, smooth_adv_0_255)

    return smooth_adv_0_255, L0_after, L0_before, L2_after, L2_before, np.asarray(
        restored_pixel_by_prediction)


def smooth_vet_can_step_adaptive(ori, adv, dnn, target_label, initial_step, strategy):
    restored_pixel_arr = []
    L0 = []
    L2 = []
    smooth_adv_0_1 = adv.reshape(-1)

    smooth_adv_0_255 = None
    for idx in range(0, 5):
        smooth_adv_0_255, L0_after, L0_before, L2_after, L2_before, restored_pixel = \
            smooth_vet_can_stepV2(ori, smooth_adv_0_1, dnn, target_label, initial_step, strategy)

        L0.append(L0_before)
        L0.append(L0_after)
        L2.append(L2_before)
        L2.append(L2_after)

        if len(restored_pixel_arr) >= 1:
            latest = restored_pixel_arr[-1]
        else:
            latest = 0
        for jdx in restored_pixel:
            restored_pixel_arr.append(jdx + latest)

        initial_step = int(np.round(initial_step / 2))
        if initial_step == 0:
            break
        else:
            smooth_adv_0_1 = smooth_adv_0_255 / 255

    restored_pixel_arr = np.asarray(restored_pixel_arr)

    return smooth_adv_0_255, L0[-1], L0[0], L2[-1], L2[0], restored_pixel_arr


#
def smooth_adv_border_V3(classifier, generated_advs, origin_images, border_indexs, target_label, step=1, K=784):
    result = []
    ranking_strategy = 'jsma'
    for adv, ori in zip(generated_advs, origin_images):
        smooth_adv, L0_after, L0_before, L2_after, L2_before, restored_pixel_by_prediction = \
            smooth_vet_can_step_adaptive(
                ori, adv, classifier,
                target_label,
                step,
                ranking_strategy)
        per_pixel_by_prediction = restored_pixel_by_prediction / L0_before
        per_pixel_by_prediction = padding_to_array(per_pixel_by_prediction, K)
        result.append(per_pixel_by_prediction)
    return np.average(result, axis=0)


def get_important_pixel_vetcan_all_images(images, classifier, path, shared_file_name):
    important_pixels_arr = []
    score_arr = []

    for image in images:
        important_pixels, score = feature_ranker.get_important_pixel_vetcan(image, classifier)
        important_pixels_arr.append(important_pixels)
        score_arr.append(score)
    important_pixels_arr_path = os.path.join(path, shared_file_name + 'pixels.npy')
    np.save(important_pixels_arr_path, np.array(important_pixels_arr))
    score_arr_path = os.path.join(path, shared_file_name + 'score.npy')
    np.save(score_arr_path, np.array(score_arr))
    return np.array(important_pixels_arr), np.array(score_arr)


def padding_to_array(arr, max_len):
    return np.concatenate((arr, [arr[-1]] * (max_len - len(arr))), axis=0)


def get_all_same_element_by_index(arr_diff, arr_value):
    if len(arr_diff) == 0:
        return None, None, None
    if len(arr_diff) == 1:
        return np.array([arr_diff[0]], dtype=np.int32), None, None

    head = arr_value[0]
    result = []
    stop_index = 0
    for index, (diff_index, diff_value) in enumerate(zip(arr_diff, arr_value)):
        if diff_value == head:
            result.append(diff_index)
        else:
            stop_index = index
            break
    return np.array(result, dtype=np.int32), arr_diff[stop_index:], arr_value[stop_index:]
