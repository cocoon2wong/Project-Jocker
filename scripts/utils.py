"""
@Author: Conghao Wong
@Date: 2022-11-11 09:28:52
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-11 09:34:19
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np


def get_value(key: str, args: list[str]):
    """
    `key` is started with `--`.
    For example, `--logs`.
    """
    args = np.array(args)
    index = np.where(args == key)[0][0]
    return str(args[index+1])
