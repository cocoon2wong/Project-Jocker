"""
@Author: Conghao Wong
@Date: 2022-09-01 10:38:40
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-01 16:02:37
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from .__base import _BaseProcess


class Move(_BaseProcess):
    """
    Move a specific point to (0, 0) according to the reference time step.
    Default reference time step is the last obsetvation step.
    """

    def __init__(self, anntype: str = None, ref: int = -1):
        super().__init__(anntype, ref)

    def preprocess(self, trajs: tf.Tensor, use_new_paras=True) -> tf.Tensor:
        """
        Move a specific point to (0, 0) according to the reference time step.

        :param trajs: trajectories, shape = `[(batch,) (K,) obs, dim]`
        :return trajs_moved: moved trajectories
        """
        if use_new_paras:
            # (batch, 1, dim)
            ref = tf.math.mod(self.ref, trajs.shape[-2])
            ref_point = tf.gather(trajs, [ref], axis=-2)
            self.paras = ref_point
        else:
            ref_point = self.paras

        if trajs.ndim == 4:   # (batch, K, obs, dim)
            ref_point = ref_point[:, tf.newaxis, :, :]

        trajs_moved = trajs - ref_point
        return trajs_moved

    def postprocess(self, trajs: tf.Tensor) -> tf.Tensor:
        """
        Move trajectories back to their original positions.

        :param trajs: trajectories moved to (0, 0) with reference point, \
            shape = `[(batch,) (K,) pred, dim]`
        :return trajs_moved: moved trajectories
        """
        # (batch, 1, dim)
        ref_point = self.paras

        while trajs.ndim > ref_point.ndim:
            ref_point = tf.expand_dims(ref_point, -3)

        trajs_moved = trajs + ref_point
        return trajs_moved
