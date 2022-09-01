"""
@Author: Conghao Wong
@Date: 2022-09-01 10:40:50
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-01 16:06:04
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...utils import SCALE_THRESHOLD
from .__base import _BaseProcess


class Scale(_BaseProcess):
    """
    Scale length of trajectories' direction vector into 1.
    Reference point when scale is the `last` observation point.
    """

    def __init__(self, anntype: str, ref: int = -1):
        super().__init__(anntype, ref)

    def preprocess(self, trajs: tf.Tensor, use_new_paras=True) -> tf.Tensor:
        """
        Scale length of trajectories' direction vector into 1.
        Reference point when scale is the `last` observation point.

        :param trajs: input trajectories, shape = `[(batch,) obs, 2]`
        :return trajs_scaled: scaled trajectories
        """
        reshape = False
        if len(trajs.shape) == 2:
            trajs = tf.expand_dims(trajs, 0)    # change into [batch, obs, 2]
            reshape = True

        if use_new_paras:
            # (batch, n)
            steps = trajs.shape[-2]
            vectors = (tf.gather(trajs, steps-1, axis=-2) - 
                tf.gather(trajs, 0, axis=-2))

            scales = []
            for [x, y] in self.order:
                vector = tf.gather(vectors, [x, y], axis=-1)
                scale = tf.linalg.norm(vector, axis=-1)
                scale = tf.maximum(SCALE_THRESHOLD, scale)

                # reshape into (batch, 1, 1)
                while scale.ndim < 3:
                    scale = tf.expand_dims(scale, -1)
                scales.append(scale)

            self.paras = scales

        else:
            scales = self.paras

        trajs_scaled = []
        steps = trajs.shape[-2]
        for scale, [x, y] in zip(scales, self.order):
            _trajs = tf.gather(trajs, [x, y], axis=-1)
            ref = tf.math.mod(self.ref, steps)
            ref_point = tf.gather(_trajs, [ref], axis=-2)
            _trajs_scaled = (_trajs - ref_point) / scale + ref_point
            trajs_scaled.append(_trajs_scaled)

        trajs_scaled = tf.concat(trajs_scaled, axis=-1)

        if reshape:
            trajs_scaled = trajs_scaled[0]

        return trajs_scaled

    def postprocess(self, trajs: tf.Tensor) -> tf.Tensor:
        """
        Scale trajectories back to their original.
        Reference point is the `first` prediction point.

        :param trajs: trajectories, shape = `[(batch,) (K,) pred, 2]`
        :param para_dict: a dict of used parameters, contains `scale:tf.Tensor`
        :return trajs_scaled: scaled trajectories
        """
        # reshape into [batch, K, pred, 2]
        expands = 0
        while trajs.ndim < 4:
            expands += 1
            trajs = tf.expand_dims(trajs, axis=-3)

        trajs_scaled = []
        scales = self.paras
        for scale, [x, y] in zip(scales, self.order):
            while scale.ndim < 4:
                scale = tf.expand_dims(scale, -1)

            _trajs = tf.gather(trajs, [x, y], axis=-1)
            _ref_point = tf.gather(_trajs, [0], axis=-2)
            _trajs_scaled = (_trajs - _ref_point) * scale + _ref_point
            trajs_scaled.append(_trajs_scaled)

        trajs_scaled = tf.concat(trajs_scaled, axis=-1)

        for _ in range(expands):
            trajs_scaled = trajs_scaled[0]

        return trajs_scaled
        