"""
@Author: Conghao Wong
@Date: 2022-07-20 10:11:08
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-20 10:11:21
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os


def dir_check(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path
