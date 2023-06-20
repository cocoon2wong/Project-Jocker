"""
@Author: Conghao Wong
@Date: 2023-06-19 20:14:39
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-20 09:20:29
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.training import loss


class keyl2(loss.l2):
    """
    l2 loss on the future keypoints.
    Support M-dimensional trajectories.
    """

    def call(self, outputs: list, labels: list, inputs: list,
             mask=None, training=None, *args, **kwargs):

        indices = self.manager.manager.model.key_indices_future
        labels_pickled = tf.gather(labels[0], indices, axis=-2)

        return super().call(outputs, [labels_pickled],
                            inputs, mask, training, *args, **kwargs)


class keyl2_past(loss.l2):
    """
    l2 loss on the future keypoints.
    Support M-dimensional trajectories.
    """

    def call(self, outputs: list, labels: list, inputs: list,
             mask=None, training=None, *args, **kwargs):

        if (indices := self.manager.manager.model.key_indices_past) is None:
            return 0

        labels_pickled = tf.gather(outputs[1], indices, axis=-2)
        return super().call([outputs[2]], [labels_pickled],
                            inputs, mask, training, *args, **kwargs)


class avgKey(loss.ADE):
    """
    l2 (2D-point-wise) loss on the future keypoints.
    """

    def call(self, outputs: list, labels: list, inputs: list,
             mask=None, training=None, *args, **kwargs):

        pred = outputs[0]
        label = labels[0]

        if pred.ndim == label.ndim:
            pred = pred[..., tf.newaxis, :, :]

        indices = self.manager.manager.model.key_indices_future
        if pred.shape[-2] != len(indices):
            pred = tf.gather(pred, indices, axis=-2)

        labels_key = tf.gather(label, indices, axis=-2)

        return super().call([pred], [labels_key], inputs,
                            mask, training, *args, **kwargs)
