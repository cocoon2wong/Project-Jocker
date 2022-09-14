"""
@Author: Conghao Wong
@Date: 2022-09-14 10:38:00
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-14 15:21:41
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import layers

from ..__args import HandlerArgs
from .__baseHandler import BaseHandlerModel


class LinearHandlerModel(BaseHandlerModel):
    """
    The Linear-Second-Stage Subnetwork
    """

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, feature_dim, points,
                         asHandler, key_points, structure,
                         *args, **kwargs)

        self.args._set('T', 'none')
        self.set_inputs('obs', 'gt')
        self.set_preprocess()
        self.linear = layers.LinearInterpolation()

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None):

        # Unpack inputs
        trajs = inputs[0]

        # Concat keypoints with the last observed point
        keypoints_index = tf.concat([[-1], keypoints_index], axis=0)
        keypoints = tf.concat([trajs[:, -1:, :], keypoints], axis=1)

        # Calculate linear interpolation -> (batch, pred, 2)
        linear = self.linear.call(keypoints_index, keypoints)

        return linear
