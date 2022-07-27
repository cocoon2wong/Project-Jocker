"""
@Author: Conghao Wong
@Date: 2022-06-29 15:36:47
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-27 20:05:06
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import plistlib

import numpy as np

from utils import dir_check

# Dataset info
DATASET = 'SDD_ped'
SPLIT_NAME = 'sdd_ped'
PROCESSED_FILE = 'ann_ped.csv'
TYPE = 'pixel'
SCALE = 100.0

# Annotation paths
SOURCE_FILE = './dataset_original/sdd/{}/video{}/annotations.txt'
TARGET_FILE = './dataset_processed/' + DATASET + '/{}/video{}/' + PROCESSED_FILE

# Saving paths
BASE_DIR = dir_check('./dataset_configs')
CURRENT_DIR = dir_check(os.path.join(BASE_DIR, DATASET))
SUBSETS_DIR = dir_check(os.path.join(CURRENT_DIR, 'subsets'))

# Dataset configs
SDD_SETS = {
    'quad':   [[0, 1, 2, 3], SCALE],
    'little':   [[0, 1, 2, 3], SCALE],
    'deathCircle':   [[0, 1, 2, 3, 4], SCALE],
    'hyang':   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], SCALE],
    'nexus':   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], SCALE],
    'coupa':   [[0, 1, 2, 3], SCALE],
    'bookstore':   [[0, 1, 2, 3, 4, 5, 6], SCALE],
    'gates':   [[0, 1, 2, 3, 4, 5, 6, 7, 8], SCALE],
}

SDD_TEST_SETS = ['coupa0',
                 'coupa1',
                 'gates2',
                 'hyang0',
                 'hyang1',
                 'hyang3',
                 'hyang8',
                 'little0',
                 'little1',
                 'little2',
                 'little3',
                 'nexus5',
                 'nexus6',
                 'quad0',
                 'quad1',
                 'quad2',
                 'quad3', ]

SDD_TRAIN_SETS = ['bookstore0',
                  'bookstore1',
                  'bookstore2',
                  'bookstore3',
                  'coupa3',
                  'deathCircle0',
                  'deathCircle1',
                  'deathCircle2',
                  'deathCircle3',
                  'deathCircle4',
                  'gates0',
                  'gates1',
                  'gates3',
                  'gates4',
                  'gates5',
                  'gates6',
                  'gates7',
                  'gates8',
                  'hyang4',
                  'hyang5',
                  'hyang6',
                  'hyang7',
                  'hyang9',
                  'nexus0',
                  'nexus1',
                  'nexus3',
                  'nexus4',
                  'nexus7',
                  'nexus8',
                  'nexus9', ]


def write_plist(value: dict, path: str):
    with open(path, 'wb+') as f:
        plistlib.dump(value, f)


def transform_annotations():
    """"
    Transform annotations with the new `ann.csv` type.
    """
    for name, value in SDD_SETS.items():
        clip_ids, scale = value
        for clip_id in clip_ids:
            source = SOURCE_FILE.format(name, clip_id)
            target = TARGET_FILE.format(name, clip_id)

            d = target.split(PROCESSED_FILE)[0]
            if not os.path.exists(d):
                os.makedirs(d)

            dat = []
            with open(source, 'r') as f:
                while data_original := f.readline():
                    data_original = data_original[:-1]
                    split = data_original.split(' ')
                    if split[6] == '1':
                        continue

                    if not split[-1] == '"Pedestrian"':
                        continue

                    dat.append([split[5],                   # frame number
                                # name of the agent
                                split[0] + '_' + split[-1],
                                float(split[2])/scale,
                                float(split[1])/scale,
                                float(split[4])/scale,
                                float(split[3])/scale, ])

            dat = np.array(dat, dtype=str)
            with open(target, 'w+') as f:
                f.writelines([','.join(item)+'\n' for item in dat])
            print('{} Done.'.format(target))


def save_dataset_info():
    """
    Save dataset information into `plist` files.
    """
    subsets = {}
    for base_set in SDD_SETS.keys():
        for index in SDD_SETS[base_set][0]:
            subsets['{}_ped_{}'.format(base_set, index)] = dict(
                name='{}_ped_{}'.format(base_set, index),
                annpath=TARGET_FILE.format(base_set, index),
                order=[0, 1],
                paras=[1, 30],
                video_path='./videos/sdd_{}_{}.mov'.format(
                    base_set, index),
                scale=SCALE,
                scale_vis=2,
                dimension=4,
                anntype='boundingbox',
                matrix=[1.0, 0.0, 1.0, 0.0],
            )

    train_sets = []
    test_sets = []
    val_sets = []

    for d in subsets.keys():
        old_name = ''.join(d.split('_ped_'))
        if old_name in SDD_TEST_SETS:
            test_sets.append(d)
        elif old_name in SDD_TRAIN_SETS:
            train_sets.append(d)
        else:
            val_sets.append(d)

    dataset_dic = dict(train=train_sets,
                       test=test_sets,
                       val=val_sets,
                       dataset=DATASET,
                       scale=SCALE,
                       dimension=4,
                       anntype='boundingbox',
                       type=TYPE)

    write_plist(dataset_dic,
                os.path.join(CURRENT_DIR, '{}.plist'.format(SPLIT_NAME)))

    for key, value in subsets.items():
        write_plist(value,
                    p := os.path.join(SUBSETS_DIR, '{}.plist'.format(key)))
        print('Successfully saved at {}'.format(p))


if __name__ == '__main__':
    transform_annotations()
    save_dataset_info()
