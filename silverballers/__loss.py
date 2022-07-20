"""
@Author: Conghao Wong
@Date: 2022-07-20 14:51:51
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-20 16:28:11
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.training import loss

from .__args import AgentArgs


class SilverballersLoss():
    """
    Structure to manage all silverballers loss and metrics.
    """

    def __init__(self, args: AgentArgs):
        self.args = args
        self.order = self.get_order()

    @property
    def p_index(self) -> tf.Tensor:
        """
        Time step of predicted key points.
        """
        p_index = [int(i) for i in self.args.key_points.split('_')]
        return tf.cast(p_index, tf.int32)

    @property
    def p_len(self) -> int:
        """
        Length of predicted key points.
        """
        return len(self.p_index)

    def get_order(self) -> list[tuple[int, int]]:
        if self.args.anntype == 'coordinate':
            order = [[0, 1]]
        elif self.args.anntype == 'boundingbox':
            order = [[0, 1], [2, 3]]
        else:
            raise NotImplementedError(self.args.anntype)
        return order

    def l2(self, outputs, labels, *args, **kwargs):
        labels_pickled = tf.gather(labels, self.p_index, axis=1)
        return loss.ADE(outputs[0], labels_pickled)

    def avgKey(self, outputs, labels, *args, **kwargs):
        labels_key = tf.gather(labels, self.p_index, axis=-2)
        return self.avgADE(outputs, labels_key)

    def avgADE(self, outputs, labels, *args, **kwargs):
        pred = outputs[0]
        order = self.order

        ade = []
        for [x, y] in order:
            _pred = tf.gather(pred, [x, y], axis=-1)
            _labels = tf.gather(labels, [x, y], axis=-1)
            ade.append(loss.ADE(_pred, _labels))

        return tf.reduce_mean(ade)

    def avgFDE(self, outputs, labels, *args, **kwargs):
        pred = outputs[0]

        pred_final = pred[:, :, -1:, :]
        labels_final = labels[:, -1:, :]

        return self.avgADE([pred_final], labels_final)
