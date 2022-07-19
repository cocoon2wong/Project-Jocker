"""
@Author: Conghao Wong
@Date: 2022-06-29 15:36:47
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-19 13:52:26
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import os
import plistlib

SCALE = 1920.0

SOURCE_FILE = './data/sdd/{}/video{}/annotations.txt'
TARGET_FILE = './data/sdd/{}/video{}/ann.csv'
BASE_DIR = './dataset_configs'

DATASET = 'SDD'
TYPE = 'pixel'


def dir_check(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


dir_check(BASE_DIR)
CURRENT_DIR = dir_check(os.path.join(BASE_DIR, DATASET))
SUBSETS_DIR = dir_check(os.path.join(CURRENT_DIR, 'subsets'))


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

SDD_TEST_SETS = ['hyang7',
                 'hyang11',
                 'bookstore6',
                 'nexus3',
                 'deathCircle4',
                 'hyang6',
                 'hyang3',
                 'little1',
                 'hyang13',
                 'gates8',
                 'gates7',
                 'hyang2']

SDD_VAL_SETS = ['nexus7',
                'coupa1',
                'gates4',
                'little2',
                'bookstore3',
                'little3',
                'nexus4',
                'hyang4',
                'gates3',
                'quad2',
                'gates1',
                'hyang9']


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

            dat = []
            with open(source, 'r') as f:
                while data_original := f.readline():
                    split = data_original.split(' ')
                    if split[6] == '1':
                        continue

                    dat.append([split[5],
                                split[0],
                                float(split[1])/scale,
                                float(split[2])/scale,
                                float(split[3])/scale,
                                float(split[4])/scale, ])

            dat = np.array(dat, dtype=np.str)
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
            subsets['{}{}'.format(base_set, index)] = dict(
                name='{}{}'.format(base_set, index),
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
        if d in SDD_TEST_SETS:
            test_sets.append(d)
        elif d in SDD_VAL_SETS:
            val_sets.append(d)
        else:
            train_sets.append(d)

    write_plist({'train': train_sets,
                 'test': test_sets,
                 'val': val_sets,
                 'dataset': DATASET,
                 'scale': SCALE,
                 'dimension': 4,
                 'anntype': 'boundingbox',
                 'type': TYPE},
                os.path.join(CURRENT_DIR, 'sdd.plist'))

    for key, value in subsets.items():
        write_plist(value,
                    p := os.path.join(SUBSETS_DIR, '{}.plist'.format(key)))
        print('Successfully saved at {}'.format(p))


if __name__ == '__main__':
    # transform_annotations()
    save_dataset_info()
