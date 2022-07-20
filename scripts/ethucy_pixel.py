"""
@Author: Conghao Wong
@Date: 2022-07-15 14:45:57
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-20 10:20:24
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import plistlib
from typing import Any

import numpy as np

from utils import dir_check

# Dataset info
DATASET = 'ETH-UCY-pixel'
TYPE = 'pixel'
SCALE = 30.0

# Annotation paths
SOURCE_FILE = './data/ethucy/{}/true_pos_.csv'
TARGET_FILE = './data/ethucy/{}/ann_pixel.csv'

# Saving paths
BASE_DIR = dir_check('./dataset_configs')
CURRENT_DIR = dir_check(os.path.join(BASE_DIR, DATASET))
SUBSETS_DIR = dir_check(os.path.join(CURRENT_DIR, 'subsets'))

# Dataset configs
SUBSETS: dict[str, Any] = {}

SUBSETS['eth_pixel'] = dict(
    name='eth_pixel',
    annpath=TARGET_FILE.format('eth'),
    order=[0, 1],
    paras=[6, 25],
    video_path='./videos/eth.mp4',
    weights=[17.667, 190.19, 10.338, 225.89],
    scale=1,
)

SUBSETS['hotel_pixel'] = dict(
    name='hotel_pixel',
    annpath=TARGET_FILE.format('hotel'),
    order=[0, 1],
    paras=[10, 25],
    video_path='./videos/hotel.mp4',
    weights=[44.788, 310.07, 48.308, 497.08],
    scale=1,
)

SUBSETS['zara1_pixel'] = dict(
    name='zara1_pixel',
    annpath=TARGET_FILE.format('zara1'),
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/zara1.mp4',
    weights=[47.707, 0, -49.727, 610.35],
    scale=1,
)

SUBSETS['zara2_pixel'] = dict(
    name='zara2_pixel',
    annpath=TARGET_FILE.format('zara2'),
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/zara2.mp4',
    weights=[47.707, 0, -49.727, 610.35],
    scale=1,
)

SUBSETS['univ_pixel'] = dict(
    name='univ_pixel',
    annpath=TARGET_FILE.format('univ'),
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/students003.mp4',
    weights=[48.607, -9.439, -41.453, 576.61],
    scale=1,
)

SUBSETS['zara3_pixel'] = dict(
    name='zara3_pixel',
    annpath=TARGET_FILE.format('zara3'),
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/zara2.mp4',
    weights=[47.707, 0, -49.727, 610.35],
    scale=1,
)

SUBSETS['univ3_pixel'] = dict(
    name='univ3_pixel',
    annpath=TARGET_FILE.format('univ3'),
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/students003.mp4',
    weights=[48.607, -9.439, -41.453, 576.61],
    scale=1,
)

SUBSETS['unive_pixel'] = dict(
    name='unive_pixel',
    annpath=TARGET_FILE.format('unive'),
    order=[1, 0],
    paras=[10, 25],
    video_path='./videos/students003.mp4',
    weights=[48.607, -9.439, -41.453, 576.61],
    scale=1,
)

TESTSETS = ['eth_pixel', 'hotel_pixel',
            'zara1_pixel', 'zara2_pixel', 'univ_pixel']


def write_plist(value: dict, path: str):
    with open(path, 'wb+') as f:
        plistlib.dump(value, f)


def transform_annotations():
    """"
    Transform annotations with the new `ann.csv` type.
    """
    for name in SUBSETS.keys():

        sname = name.split('_pixel')[0]
        source = SOURCE_FILE.format(sname)
        target = TARGET_FILE.format(sname)

        data_original = np.loadtxt(source, delimiter=',')
        r = data_original[2:].T

        weights = SUBSETS[name]['weights']
        order = SUBSETS[name]['order']

        result = np.column_stack([
            weights[2] * r.T[1] + weights[3],
            weights[0] * r.T[0] + weights[1],
        ])/SCALE

        dat = np.column_stack([data_original[0].astype(int).astype(str),
                               data_original[1].astype(int).astype(str),
                               result.T[order[0]].astype(str),
                               result.T[order[1]].astype(str)])

        with open(target, 'w+') as f:
            for _dat in dat:
                f.writelines([','.join(_dat)+'\n'])
        print('{} Done.'.format(target))


def save_dataset_info():
    """
    Save dataset information into `plist` files.
    """
    subsets = {}
    for name, value in SUBSETS.items():
        subsets[name] = dict(
            name=name,
            annpath=value['annpath'],
            order=[1, 0],
            paras=value['paras'],
            video_path=value['video_path'],
            scale=SCALE,
            scale_vis=1,
            dimension=2,
            anntype='coordinate',
            matrix=[1.0, 0.0, 1.0, 0.0],
        )

    for ds in TESTSETS:
        train_sets = []
        test_sets = []
        val_sets = []

        for d in subsets.keys():
            if d == ds:
                test_sets.append(d)
                val_sets.append(d)
            else:
                train_sets.append(d)

        dataset_dic = dict(train=train_sets,
                           test=test_sets,
                           val=val_sets,
                           dataset=DATASET,
                           scale=SCALE,
                           dimension=2,
                           anntype='coordinate',
                           type=TYPE)

        write_plist(dataset_dic,
                    os.path.join(CURRENT_DIR, '{}.plist'.format(ds)))

    for key, value in subsets.items():
        write_plist(value,
                    p := os.path.join(SUBSETS_DIR, '{}.plist'.format(key)))
        print('Successfully saved at {}'.format(p))


if __name__ == '__main__':
    transform_annotations()
    save_dataset_info()
