"""
Created At: 14/07/2021 14:28
"""
import os
from datetime import datetime

import cv2
import numpy as np
import sys


def get_timestamp(timestamp_format="%d%m%d-%H%M%S"):
    return datetime.now().strftime(timestamp_format)


def check_path_exists(path):
    return os.path.exists(path)


def mkdir(path):
    if not check_path_exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def check_file_extension(file_path: str, extension_type: str):
    return file_path.endswith(extension_type)


def get_file_name(file_path: str):
    full_file_name = os.path.basename(file_path)
    return os.path.splitext(full_file_name)[0]


# image

def get_border(image: np.ndarray) -> np.ndarray:
    min = np.min(image)
    max = np.max(image)
    img_0_255 = image.astype(np.unit8)
    if min >= 0. and max <= 1.:
        img_0_255 = (image * 255.).astype(np.unit8)
    border_img = np.array(cv2.Canny(img_0_255, 100, 200)).reshape((image.shape[0], image.shape[1], 1))
    return border_img


def get_borders(images: np.ndarray) -> np.ndarray:
    border_results = []
    for image in images:
        border_results.append(get_border(image))
    return np.array(border_results, dtype=np.float32) / 255.


def write_to_file(content, path: str, mode='w'):
    file_writer = open(file=path, mode=mode)
    file_writer.write(content)
    file_writer.close()


# metric

def compute_l0(adv: np.ndarray,
               ori: np.ndarray,
               normalized=False):  # 1d array, value in range of [0 .. 1]
    if not normalized:
        adv = np.round(adv * 255)
        ori = np.round(ori * 255)
    adv = adv.reshape(-1)
    ori = ori.reshape(-1)
    l0_dist = 0
    for idx in range(len(adv)):
        if adv[idx] != ori[idx]:
            l0_dist += 1
    return l0_dist


def compute_distance(data_1: np.ndarray, data_2: np.ndarray):
    result_l0, result_l2 = [], []
    for data_1_i, data_2_i in zip(data_1, data_2):
        result_l0.append(compute_l0(data_1_i, data_2_i))
        result_l2.append(compute_l2(data_1_i, data_2_i))

    return np.asarray(result_l0), np.asarray(result_l2)


def compute_l2(adv: np.ndarray,
               ori: np.ndarray):
    if not (np.min(adv) >= 0 and np.max(adv) <= 1):
        adv = adv / 255
    if not (np.min(ori) >= 0 and np.max(ori) <= 1):
        ori = ori / 255
    return np.linalg.norm(adv.reshape(-1) - ori.reshape(-1))


def exit_execution(msg: str):
    sys.exit(msg)

