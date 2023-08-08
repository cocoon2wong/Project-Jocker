"""
@Author: Conghao Wong
@Date: 2023-08-08 14:55:56
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-08 16:44:39
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf

from codes.utils import get_mask


class SocialCircleLayer(tf.keras.layers.Layer):

    def __init__(self, partitions: int,
                 max_partitions: int = None,
                 relative_velocity=True,
                 mu=0.0001,
                 *args, **kwargs):
        """
        A layer to compute the SocialCircle.

        :param partitions: The number of partitions in the circle.
        :param relative_velocity: Choose whether to use relative velocity or not.
        :param mu: The small number to prevent dividing zero when computing.
        """
        super().__init__(*args, **kwargs)

        self.partitions = partitions
        self.max_partitions = max_partitions
        self.rel_velocity = relative_velocity
        self.mu = mu

    def call(self, trajs, nei_trajs, *args, **kwargs):
        # Move vectors -> (batch, ..., 2)
        obs_vector = trajs[..., -1:, :] - trajs[..., 0:1, :]
        nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]

        # Calculate velocities
        nei_vector_len = tf.linalg.norm(nei_vector, axis=-1)    # (batch, n)
        obs_vector_len = tf.linalg.norm(obs_vector, axis=-1)    # (batch, 1)

        # Speed factor in the SocialCircle
        if self.rel_velocity:
            f_speed = (nei_vector_len + self.mu) / (obs_vector_len + self.mu)
        else:
            f_speed = nei_vector_len

        # Distance factor
        nei_posion_vector = (nei_trajs[..., -1, :] -
                             trajs[..., tf.newaxis, -1, :])
        f_distance = tf.linalg.norm(nei_posion_vector, axis=-1)

        # Direction factor
        f_direction = tf.atan2(x=nei_posion_vector[..., 0],
                               y=nei_posion_vector[..., 1])
        f_direction = tf.math.mod(f_direction, 2*np.pi)

        # Angles (the independent variable \theta)
        angle_indices = f_direction / (2*np.pi/self.partitions)
        angle_indices = tf.cast(angle_indices, tf.int32)

        # Mask neighbors
        nei_mask = get_mask(tf.reduce_sum(nei_trajs, axis=[-1, -2]), tf.int32)
        angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

        # Compute the SocialCircle
        social_circle = []
        for ang in range(self.partitions):
            _mask = tf.cast(angle_indices == ang, tf.float32)
            _mask_count = tf.reduce_sum(_mask, axis=-1)

            n = _mask_count + 0.0001
            _speed = tf.reduce_sum(f_speed * _mask, axis=-1) / n
            _distance = tf.reduce_sum(f_distance * _mask, axis=-1) / n
            _direction = tf.reduce_sum(f_direction * _mask, axis=-1) / n
            social_circle.append([_speed, _distance, _direction])

        # Shape of the final SocialCircle: (batch, p, 3)
        social_circle = tf.cast(social_circle, tf.float32)
        social_circle = tf.transpose(social_circle, [2, 0, 1])

        if (((m := self.max_partitions) is not None) and
                (m > (n := self.partitions))):
            paddings = tf.constant([[0, 0], [0, m - n], [0, 0]])
            social_circle = tf.pad(social_circle, paddings)

        return social_circle, f_direction
