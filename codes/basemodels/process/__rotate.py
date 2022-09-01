"""
@Author: Conghao Wong
@Date: 2022-09-01 11:15:52
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-01 16:55:38
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...utils import ROTATE_BIAS
from .__base import _BaseProcess


class Rotate(_BaseProcess):
    """
    Rotate trajectories to the referce angle.
    Default reference angle is 0.
    """

    def __init__(self, anntype: str, ref):
        super().__init__(anntype, ref)

    def preprocess(self, trajs: tf.Tensor, use_new_paras=True) -> tf.Tensor:
        """
        Rotate trajectories to the referce angle.

        :param trajs: observations, shape = `[(batch,) obs, dim]`
        :return trajs_rotated: moved trajectories
        """
        if use_new_paras:
            # (batch, dim)
            steps = trajs.shape[-2]
            vectors = (tf.gather(trajs, steps-1, axis=-2) -
                       tf.gather(trajs, 0, axis=-2))

            angles = []
            for [x, y] in self.order:
                vector_x = tf.gather(vectors, x, axis=-1)
                vector_y = tf.gather(vectors, y, axis=-1)
                main_angle = tf.atan((vector_y + ROTATE_BIAS) /
                                     (vector_x + ROTATE_BIAS))
                angle = self.ref - main_angle
                angles.append(angle)

            self.paras = angles

        else:
            angles = self.paras

        trajs_rotated = []
        for angle, [x, y] in zip(angles, self.order):
            rotate_matrix = tf.stack([[tf.cos(angle), tf.sin(angle)],
                                      [-tf.sin(angle), tf.cos(angle)]])

            if rotate_matrix.ndim == 3:
                # transpose to (batch, 2, 2)
                rotate_matrix = tf.transpose(rotate_matrix, [2, 0, 1])

            _trajs = tf.gather(trajs, [x, y], axis=-1)
            _trajs_rotated = _trajs @ rotate_matrix
            trajs_rotated.append(_trajs_rotated)

        trajs_rotated = tf.concat(trajs_rotated, axis=-1)
        return trajs_rotated

    def postprocess(self, trajs: tf.Tensor) -> tf.Tensor:
        """
        Rotate trajectories back to their original angles.

        :param trajs: trajectories, shape = `[(batch, ) pred, dim]`
        :return trajs_rotated: rotated trajectories
        """
        angles = self.paras
        S = tf.cast(trajs.shape, tf.int32)

        trajs_rotated = []
        for angle, [x, y] in zip(angles, self.order):
            angle = -1 * angle
            rotate_matrix = tf.stack([[tf.cos(angle), tf.sin(angle)],
                                      [-tf.sin(angle), tf.cos(angle)]])

            _trajs = tf.gather(trajs, [x, y], axis=-1)

            if len(S) >= 3:
                # traj shape = (batch, pred, 2)
                rotate_matrix = tf.transpose(rotate_matrix, [2, 0, 1])

            if len(S) == 4:
                # traj shape = (batch, K, pred, 2)
                _trajs = tf.reshape(_trajs, (S[0]*S[1], S[2], -1))
                rotate_matrix = tf.repeat(rotate_matrix, S[1], axis=0)

            _trajs_rotated = _trajs @ rotate_matrix
            trajs_rotated.append(_trajs_rotated)

        trajs_rotated = tf.concat(trajs_rotated, axis=-1)

        if len(S) == 4:
            trajs_rotated = tf.reshape(trajs_rotated, S)

        return trajs_rotated
