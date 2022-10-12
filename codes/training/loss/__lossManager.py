"""
@Author: Conghao Wong
@Date: 2022-10-12 11:13:46
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-12 19:12:10
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Any

import tensorflow as tf

from ...base import BaseManager
from .__ade import ADE
from .__iou import AIoU, FIoU


class LossManager(BaseManager):

    def __init__(self, name: str,
                 manager: BaseManager):

        super().__init__(manager=manager)

        self.name = name

        self.loss_list = []
        self.loss_weights = []
        self.metrics_list = []
        self.metrics_weights = []

        if self.args.anntype == 'coordinate':
            self.order = [[0, 1]]
        elif self.args.anntype == 'boundingbox':
            self.order = [[0, 1], [2, 3]]
        else:
            raise NotImplementedError(self.args.anntype)

    @property
    def p_index(self) -> tf.Tensor:
        """
        Time step of predicted key points.
        """
        if 'key_points' in self.args.__dir__():
            p_index = [int(i) for i in self.args.key_points.split('_')]
        else:
            p_index = list(range(self.args.pred_frames))

        return tf.cast(p_index, tf.int32)

    @property
    def p_len(self) -> int:
        """
        Length of predicted key points.
        """
        return len(self.p_index)

    def set(self, loss_dict: dict[Any, float]):
        self.loss_list = [k for k in loss_dict.keys()]
        self.loss_weights = [v for v in loss_dict.values()]

    def call(self, outputs: list[tf.Tensor],
             labels: tf.Tensor,
             training=None,
             coefficient: float = 1.0):

        loss_dict = {}
        for loss_func in self.loss_list:
            name = loss_func.__name__
            value = loss_func(outputs, labels,
                              coe=coefficient,
                              training=training)

            loss_dict[f'{name}({self.name})'] = value

        if (l := len(self.loss_weights)):
            if l != len(loss_dict):
                raise ValueError('Incorrect loss weights!')
            weights = self.loss_weights

        else:
            weights = tf.ones(len(loss_dict))

        summary = tf.matmul(tf.expand_dims(list(loss_dict.values()), 0),
                            tf.expand_dims(weights, 1))
        summary = tf.reshape(summary, ())
        return summary, loss_dict

    ####################################
    # Loss functions are defined below
    ####################################

    def l2(self, outputs: list[tf.Tensor],
           labels: tf.Tensor,
           coe: float = 1.0,
           *args, **kwargs):
        """
        l2 loss on the keypoints.
        Support M-dimensional trajectories.
        """
        labels_pickled = tf.gather(labels, self.p_index, axis=1)
        return ADE(outputs, labels_pickled, coe=coe)

    def avgKey(self, outputs: list[tf.Tensor],
               labels: tf.Tensor,
               coe: float = 1.0,
               *args, **kwargs):
        """
        l2 (2D-point-wise) loss on the keypoints.

        :param outputs: a list of tensors, where `outputs[0].shape`
            is `(batch, K, pred, 2)` or `(batch, pred, 2)`
            or `(batch, K, n_key, 2)` or `(batch, n_key, 2)`
        :param labels: shape is `(batch, pred, 2)`
        """
        pred = outputs[0]
        if pred.ndim == 3:
            pred = pred[:, tf.newaxis, :, :]

        if pred.shape[-2] != self.p_len:
            pred = tf.gather(pred, self.p_index, axis=-2)

        labels_key = tf.gather(labels, self.p_index, axis=-2)

        return ADE([pred], labels_key, coe)

    def ADE(self, outputs: list[tf.Tensor],
            labels: tf.Tensor,
            coe: float = 1.0,
            *args, **kwargs):
        """
        l2 (2D-point-wise) loss.
        Support M-dimensional trajectories.

        :param outputs: a list of tensors, where `outputs[0].shape` 
            is `(batch, K, pred, 2)` or `(batch, pred, 2)`
        :param labels: shape is `(batch, pred, 2)`
        """
        pred = outputs[0]
        order = self.order

        if pred.ndim == 3:
            pred = pred[:, tf.newaxis, :, :]

        ade = []
        for [x, y] in order:
            _pred = tf.gather(pred, [x, y], axis=-1)
            _labels = tf.gather(labels, [x, y], axis=-1)
            ade.append(ADE([_pred], _labels, coe))

        return tf.reduce_mean(ade)

    def FDE(self, outputs: list[tf.Tensor],
            labels: tf.Tensor,
            coe: float = 1.0,
            *args, **kwargs):
        """
        l2 (2D-point-wise) loss on the last prediction point.
        Support M-dimensional trajectories.

        :param outputs: a list of tensors, where 
            `outputs[0].shape` is `(batch, K, pred, 2)`
            or `(batch, pred, 2)`
        :param labels: shape is `(batch, pred, 2)`
        """
        pred = outputs[0]

        if pred.ndim == 3:
            pred = pred[:, tf.newaxis, :, :]

        pred_final = pred[..., -1:, :]
        labels_final = labels[..., -1:, :]

        return self.ADE([pred_final], labels_final, coe)

    def AIoU(self, outputs: list[tf.Tensor],
             labels: tf.Tensor,
             coe: float = 1.0,
             *args, **kwargs):

        return AIoU(outputs, labels, coe=1.0)

    def FIoU(self, outputs: list[tf.Tensor],
             labels: tf.Tensor,
             coe: float = 1.0,
             *args, **kwargs):

        return FIoU(outputs, labels, coe=1.0, index=-1)

    def HIoU(self, outputs: list[tf.Tensor],
             labels: tf.Tensor,
             coe: float = 1.0,
             *args, **kwargs):

        s = self.args.pred_frames
        length = 2 if s % 2 == 0 else 1
        index = s//2 - 1

        return FIoU(outputs, labels, coe=1.0,
                    index=index, length=length)
