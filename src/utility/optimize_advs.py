"""
Created At: 29/07/2021 22:32
"""

import numpy as np
import tensorflow as tf

from utility.feature_ranker import feature_ranker


# progbar = tf.keras.utils.Progbar(len(train_data))


def optimize_advs(classifier, generated_advs, origin_images, target_label, step,
                  return_adv=False, num_class=10, epoches=5):
    input_shape = generated_advs[0].shape
    total_element = np.prod(input_shape)

    advs_len = len(generated_advs)
    generated_advs = generated_advs.reshape((-1, total_element))
    origin_images = origin_images.reshape((-1, total_element))
    origin_images_0_255 = np.round(origin_images * 255)
    smooth_advs_0_255s = np.round(generated_advs * 255).reshape((-1, total_element))

    new_smooth_advs = []
    batch_size = 2000
    for epoch in range(epoches):
        print(f'epoch to optimize: {epoch + 1}/{epoches}')
        if len(new_smooth_advs) != 0:
            smooth_advs_0_255s = np.array(new_smooth_advs)
        new_smooth_advs = np.array([])
        progbar = tf.keras.utils.Progbar(advs_len)
        recover_speed = None
        for batch_index in range(0, advs_len, batch_size):
            # print(f'batch: {batch_index}')
            smooth_advs_0_255 = optimize_batch(classifier,
                                               smooth_advs_0_255s[
                                               batch_index: batch_index + batch_size] / 255.,
                                               origin_images[batch_index: batch_index + batch_size],
                                               smooth_advs_0_255s[batch_index: batch_index + batch_size],
                                               origin_images_0_255[
                                               batch_index: batch_index + batch_size], target_label,
                                               step, num_class)
            # if latest_recover_speed is None:
            #     latest_recover_speed = list(recover_speed)
            # else:
            #     for index in range(len(recover_speed)):
            #         for index_j in range(len(recover_speed[index])):
            #             latest_recover_speed[index].append(latest_recover_speed[index][-1] + recover_speed[index][index_j])
            if len(new_smooth_advs) == 0:
                new_smooth_advs = smooth_advs_0_255
            else:
                new_smooth_advs = np.concatenate((new_smooth_advs, smooth_advs_0_255))
            progbar.update(batch_index + batch_size if batch_index + batch_size < advs_len else advs_len)

        step = step // 2
        if step == 0:
            break

    return smooth_advs_0_255s.reshape(-1, *input_shape) / 255.


def optimize_batch(classifier, generated_advs, origin_images, generated_advs_0_255, origin_images_0_255, target_label,
                   step, num_class=10):
    diff_pixels = []
    batch_size = len(generated_advs)
    # smooth_advs_0_255_arrs = np.array(generated_advs_0_255)
    # diff_pixel_arrs = None
    for index in range(batch_size):
        compare = generated_advs_0_255[index] == origin_images_0_255[index]
        diff_pixels.append(np.where(compare == False)[0])
    diff_pixel_arrs, _ = feature_ranker.random_ranking_batch(generated_advs=generated_advs,
                                                                         origin_images=origin_images,
                                                                         target_label=target_label,
                                                                         classifier=classifier,
                                                                         diff_pixels=diff_pixels,
                                                                         num_class=num_class)
    # print(diff_pixel_arrs)
    smooth_advs_0_255 = recover_batch(classifier=classifier, generated_advs=generated_advs,
                                      origin_images=origin_images,
                                      generated_advs_0_255=generated_advs_0_255,
                                      origin_images_0_255=origin_images_0_255, target_label=target_label,
                                      step=step, diff_pixel_arrs=diff_pixel_arrs)
    return smooth_advs_0_255


def recover_batch(classifier, generated_advs, origin_images, generated_advs_0_255, origin_images_0_255, target_label,
                  step, diff_pixel_arrs=None):
    # make all diff_pixel arr to same length
    padded_diff_pixels = padd_to_arrs(diff_pixel_arrs, padded_value=784, d_type=int)

    # adding to new pixel for each image
    padded_generated_advs_0_255 = padd_to_arrs(generated_advs_0_255, max_length=784 + 1)
    padded_origin_images_0_255 = padd_to_arrs(origin_images_0_255, max_length=784 + 1)
    old_padded_generated_advs_0_255 = np.array(padded_generated_advs_0_255)
    max_diff_lenth = max(map(len, diff_pixel_arrs))
    for step_i in range(0, max_diff_lenth, step):
        for index in range(len(generated_advs)):
            indexes = padded_diff_pixels[index][step_i:step_i + step].astype(int)

            padded_generated_advs_0_255[index, indexes] = \
                padded_origin_images_0_255[index, indexes]

        predictions = classifier.predict(
            padded_generated_advs_0_255[:, :-1].reshape(-1, 28, 28, 1) / 255.)

        predictions = np.argmax(predictions, axis=1)
        unrecovered_adv_indexes = np.where(predictions != target_label)[0]
        # check recover effect to each prediction
        if len(unrecovered_adv_indexes) == 0:
            continue
        for prediction_index in unrecovered_adv_indexes:
            indexes = padded_diff_pixels[prediction_index][step_i:step_i + step].astype(int)
            padded_generated_advs_0_255[prediction_index, indexes] = \
                old_padded_generated_advs_0_255[prediction_index, indexes]

    # ignore the new pixel padded to image
    return padded_generated_advs_0_255[:, :-1]


def padd_to_arrs(arrs, padded_value=784, max_length=None, d_type=float):
    max_length = max(map(len, arrs)) if max_length is None else max_length
    result = []
    for arr in arrs:
        padded_arr = np.concatenate((arr, [padded_value] * (max_length - len(arr))))
        result.append(padded_arr)
    return np.asarray(result).astype(d_type)
