"""
@Author: Conghao Wong
@Date: 2022-09-14 10:38:00
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-23 20:40:58
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels import layers
from codes.constant import INPUT_TYPES

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
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)
        self.set_preprocess()
        self.linear = layers.LinearInterpolation()
        self.accept_batchK_inputs = True

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None):

        # Unpack inputs
        trajs = inputs[0]

        if keypoints.ndim == 4:     # (batch, K, steps, dim)
            K = keypoints.shape[-3]
            trajs = tf.repeat(trajs[:, tf.newaxis], K, axis=-3)

        # Concat keypoints with the last observed point
        keypoints_index = tf.concat([[-1], keypoints_index], axis=0)
        obs_position = tf.gather(trajs, [self.args.obs_frames-1], axis=-2)
        keypoints = tf.concat([obs_position, keypoints], axis=-2)

        # Calculate linear interpolation -> (batch, pred, 2)
        linear = self.linear.call(keypoints_index, keypoints)

        return linear
