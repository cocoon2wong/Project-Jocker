"""
@Author: Conghao Wong
@Date: 2022-10-12 11:13:46
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-20 10:47:33
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Any

import tensorflow as tf

from ...base import BaseManager
from ...dataset import AnnotationManager
from .__ade import ADE_2D
from .__iou import AIoU, FIoU


class LossManager(BaseManager):

    def __init__(self, manager: BaseManager, name='Loss Manager'):
        """
        Init a `LossManager` object.

        :param manager: manager object, usually a `Structure` object.
        :param name: name of the manager, which could appear in all dict
            keys in the final output `loss_dict`.
        """

        super().__init__(manager=manager, name=name)

        self.AIoU = AIoU
        self.FIoU = FIoU

        self.loss_list = []
        self.loss_weights = []

    @property
    def picker(self) -> AnnotationManager:
        return self.manager.get_member(AnnotationManager)

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
        """
        Set loss functions and their weights.

        :param loss_dict: a dict of loss functions, where all dict keys
            are the callable loss function, and the dict values are the
            weights of the corresponding loss function.
        """
        self.loss_list = [k for k in loss_dict.keys()]
        self.loss_weights = [v for v in loss_dict.values()]

    def call(self, outputs: list[tf.Tensor],
             labels: tf.Tensor,
             training=None,
             coefficient: float = 1.0):
        """
        Call all loss functions recorded in the `loss_list`.

        :param outputs: a list of model's output tensors.
        :param labels: ground truth tensor.
        :param training: choose whether to run as the training mode.
        :param coefficient: scale parameter on the loss functions.

        :return summary: the weighted sum of all loss functions.
        :return loss_dict: a dict of values of all loss functions.
        """

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
        return ADE_2D(outputs[0], labels_pickled, coe=coe)

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

        return ADE_2D(pred, labels_key, coe)

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

        if pred.ndim == 3:
            pred = pred[:, tf.newaxis, :, :]

        ade = []
        for p, gt in zip(self.picker.get_coordinate_series(pred),
                         self.picker.get_coordinate_series(labels)):
            ade.append(ADE_2D(p, gt, coe))

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

    def HIoU(self, outputs: list[tf.Tensor],
             labels: tf.Tensor,
             coe: float = 1.0,
             *args, **kwargs):

        s = self.args.pred_frames
        length = 2 if s % 2 == 0 else 1
        index = s//2 - 1

        return FIoU(outputs, labels, coe=1.0,
                    index=index, length=length)

    def print_info(self, **kwargs):
        funcs = [f.__name__ for f in self.loss_list]
        return super().print_info(Functions=funcs,
                                  Weights=self.loss_weights,
                                  **kwargs)
