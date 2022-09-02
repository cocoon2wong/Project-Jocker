"""
@Author: Conghao Wong
@Date: 2022-09-01 10:38:49
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-02 14:39:03
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import tensorflow as tf


class BasePreProcessor():

    def __init__(self, anntype: str, ref):

        self.ref = ref
        self.anntype = anntype
        self.paras = None

        self.order = self.set_order(anntype)

    def preprocess(self, trajs: tf.Tensor, use_new_paras=True) -> tf.Tensor:
        raise NotImplementedError('Please rewrite this method')

    def postprocess(self, trajs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError('Please rewrite this method')

    def update_paras(self, trajs: tf.Tensor) -> None:
        raise NotImplementedError('Please rewrite this method')

    def set_order(self, anntype: str):
        if anntype is None:
            return None

        if anntype == 'coordinate':
            order = [[0, 1]]
        elif anntype == 'boundingbox':
            order = [[0, 1], [2, 3]]
        else:
            raise NotImplementedError(anntype)

        return order


def update(new: Union[tuple, list],
           old: Union[tuple, list]) -> tuple:

    if type(old) == list:
        old = tuple(old)
    if type(new) == list:
        new = tuple(new)

    if len(new) < len(old):
        return new + old[len(new):]
    else:
        return new
        