"""
@Author: Conghao Wong
@Date: 2021-07-19 11:11:10
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-30 20:15:59
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import sys

import numpy as np


def clean_logs(base_dir):
    """
    Delete all saved model weights except the best one.
    """
    for d in os.listdir(base_dir):
        cd = os.path.join(base_dir, d)

        if d.startswith('.') or not os.path.isdir(cd):
            continue

        files = os.listdir(cd)

        if (fn := 'best_ade_epoch.txt') in files:
            best_epoch = np.loadtxt(os.path.join(cd, fn))[1].astype(int)
            pattern = '_epoch{}.tf'.format(best_epoch)

        else:
            continue

        for f in files:
            path = os.path.join(cd, f)
            if pattern in f:
                print('Find {}.'.format(path))

            else:
                if f.endswith('.tf.index') or '.tf.data' in f:
                    print('Remove {}.'.format(path))
                    os.remove(path)


def clean_figs(base_dir):
    """
    Delete all saved visualizations in the `base_dir`.
    """
    for d in os.listdir(base_dir):
        cd = os.path.join(base_dir, d)

        if d.startswith('.') or not os.path.isdir(cd):
            continue

        files = os.listdir(cd)

        if (fn := 'VisualTrajs') in files:
            os.system('rm -r {}'.format(p := os.path.join(cd, fn)))
            print('Remove {}.'.format(p))


def get_value(key: str, args: list[str]):
    """
    `key` is started with `--`.
    For example, `--logs`.
    """
    args = np.array(args)
    index = np.where(args == key)[0][0]
    return str(args[index+1])


if __name__ == '__main__':
    args = sys.argv
    if '--logs' in args:
        clean_logs(get_value('--logs', args))

    elif '--figs' in args:
        clean_figs(get_value('--figs', args))
