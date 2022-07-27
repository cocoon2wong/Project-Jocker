"""
@Author: Conghao Wong
@Date: 2022-06-20 20:10:58
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-27 13:51:47
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import plistlib
import time

"""
Configs
"""
# Basic parameters
TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
DATASET_DIR = './dataset_configs'

# Dataset configs
INIT_POSITION = 10000

# Context map configs
# WINDOW_EXPAND_PIXEL = 0.3
# WINDOW_SIZE_PIXEL = 200.0
WINDOW_EXPAND_PIXEL = 10.0
WINDOW_SIZE_PIXEL = 10.0

WINDOW_EXPAND_METER = 10.0
WINDOW_SIZE_METER = 10.0

MAP_HALF_SIZE = 50  # Local map's half size
AVOID_SIZE = 15     # Avoid size in grid cells when modeling social interaction
INTEREST_SIZE = 20  # Interest size in grid cells when modeling social interaction

# Preprocess configs
ROTATE_BIAS = 0.01
SCALE_THRESHOLD = 0.05

# Visualization configs
SMALL_POINTS = True
OBS_IMAGE = './figures/obs_small.png' if SMALL_POINTS else './figures/obs.png'
GT_IMAGE = './figures/gt_small.png' if SMALL_POINTS else './figures/gt.png'
PRED_IMAGE = './figures/pred_small.png' if SMALL_POINTS else './figures/pred.png'
DISTRIBUTION_IMAGE = './figures/dis.png'

# Log paths
TEMP_PATH = './temp_files'


def dir_check(target_dir: str) -> str:
    """
    Used for check if the `target_dir` exists.
    It not exist, it will make it.
    """
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    return target_dir


def load_from_plist(path: str) -> dict:
    """
    Load plist files into python `dict` object.

    :param path: path of the plist file
    :return dat: a `dict` object loaded from the file
    """
    with open(path, 'rb') as f:
        dat = plistlib.load(f)

    return dat
