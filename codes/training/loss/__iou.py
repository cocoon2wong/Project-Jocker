"""
@Author: Conghao Wong
@Date: 2022-10-12 10:50:35
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-12 13:12:02
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


def AIoU(outputs: list[tf.Tensor],
         GT: tf.Tensor,
         coe: float = 1.0) -> tf.Tensor:
    """
    Calculate the average IoU on predicted bounding boxes among the `time` axis.
    It is only used for models with `anntype == 'boundingbox'`.
    Each dimension of the predictions should be `(xl, yl, xr, yr)`.
    """
    pred = outputs[0]

    if pred.ndim == 3:
        pred = pred[:, tf.newaxis, :, :]

    all_iou = []
    for k in range(pred.shape[1]):
        all_iou.append(__IoU_single(pred[:, k, :, :], GT))

    all_iou = tf.stack(all_iou)     # (K, batch, steps)
    all_iou = tf.reduce_max(all_iou, axis=0)
    all_iou = tf.reduce_mean(all_iou, axis=-1)

    return tf.reduce_mean(all_iou)


def FIoU(outputs: list[tf.Tensor],
         GT: tf.Tensor,
         coe: float = 1.0) -> tf.Tensor:
    """
    Calculate the IoU on the final prediction time step.
    It is only used for models with `anntype == 'boundingbox'`.
    Each dimension of the predictions should be `(xl, yl, xr, yr)`.
    """
    pred = outputs[0]
    return AIoU([pred[..., -1:, :]], GT[..., -1:, :])


def __IoU_single(pred: tf.Tensor, GT: tf.Tensor) -> tf.Tensor:
    """
    Calculate IoU on pred and GT.
    pred and GT must have the same shape.

    :param pred: shape = (..., 4)
    :param GT: shape = (..., 4)
    """

    # (..., 8)
    s = tf.shape(pred)[:-1]
    dat = tf.concat([pred, GT], axis=-1)

    x1 = tf.gather(pred, [0, 2], axis=-1)
    x2 = tf.gather(GT, [0, 2], axis=-1)
    x1_len = tf.abs(x1[..., 0] - x1[..., 1])
    x2_len = tf.abs(x2[..., 0] - x2[..., 1])

    x_sorted = tf.sort(tf.gather(dat, [0, 2, 4, 6], axis=-1), axis=-1)
    x_len_all = tf.abs(x_sorted[..., 0] - x_sorted[..., -1])

    x_len_inter = x1_len + x2_len - x_len_all
    x_len_inter = tf.maximum(0, x_len_inter)

    y1 = tf.gather(pred, [1, 3], axis=-1)
    y2 = tf.gather(GT, [1, 3], axis=-1)
    y1_len = tf.abs(y1[..., 0] - y1[..., 1])
    y2_len = tf.abs(y2[..., 0] - y2[..., 1])

    y_sorted = tf.sort(tf.gather(dat, [1, 3, 5, 7], axis=-1), axis=-1)
    y_len_all = tf.abs(y_sorted[..., 0] - y_sorted[..., -1])

    y_len_inter = y1_len + y2_len - y_len_all
    y_len_inter = tf.maximum(0, y_len_inter)

    s_inter = x_len_inter * y_len_inter
    s_all = x1_len * y1_len + x2_len * y2_len

    iou = s_inter / (s_all - s_inter)

    return iou
