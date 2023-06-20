"""
@Author: Conghao Wong
@Date: 2022-10-12 09:06:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-20 09:03:13
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


def ADE_2D(pred: tf.Tensor,
           GT: tf.Tensor,
           coe: float = 1.0,
           mask: tf.Tensor = None) -> tf.Tensor:
    """
    Calculate `ADE` or `minADE`.

    :param pred: The predicted trajectories with shape = \
        `(batch, K, steps, 2)` or `(batch, steps, 2)`.
    :param GT: Ground truth future trajectory, shape = `(batch, steps, 2)`.

    :return loss_ade:
        Return `ADE` when input_shape = [batch, pred_frames, 2];
        Return `minADE` when input_shape = [batch, K, pred_frames, 2].
    """
    if pred.ndim == GT.ndim:      # (batch, steps, dim)
        pred = pred[..., tf.newaxis, :, :]

    # Shape of all_ade: (..., K)
    all_ade = tf.reduce_mean(
        tf.linalg.norm(
            pred - GT[..., tf.newaxis, :, :],
            ord=2, axis=-1
        ), axis=-1)

    best_ade = tf.reduce_min(all_ade, axis=-1)

    if mask is not None:
        best_ade *= mask
        count = tf.reduce_sum(mask)
    else:
        count = tf.reduce_sum(tf.ones_like(best_ade))

    return coe * tf.reduce_sum(best_ade) / count
