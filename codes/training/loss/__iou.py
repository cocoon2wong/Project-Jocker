"""
@Author: Conghao Wong
@Date: 2022-10-12 10:50:35
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-12 13:53:25
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

    K = pred.shape[1]
    GT = tf.repeat(GT[:, tf.newaxis, :, :], K, axis=-3)

    # (batch, K, steps)
    iou = __IoU_single_2Dbox(pred, GT)
    iou = tf.reduce_mean(iou, axis=-1)
    iou = tf.reduce_max(iou, axis=1)
    return tf.reduce_mean(iou)


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


def __IoU_single_2Dbox(box1: tf.Tensor, box2: tf.Tensor) -> tf.Tensor:
    """
    Calculate IoU on pred and GT.
    pred and GT must have the same shape.

    :param pred: shape = (..., 4)
    :param GT: shape = (..., 4)
    """

    # (0,   1,   2,   3,   4,   5,   6,   7  )
    # (xl1, yl1, xr1, yrl, xl2, yl2, xr2, yr2)
    dat = tf.concat([box1, box2], axis=-1)

    len1_x, len2_x, len_inter_x = \
        __get_len_1Dbox(tf.gather(dat, [0, 2], axis=-1),
                        tf.gather(dat, [4, 6], axis=-1))

    len1_y, len2_y, len_inter_y = \
        __get_len_1Dbox(tf.gather(dat, [1, 3], axis=-1),
                        tf.gather(dat, [5, 7], axis=-1))

    s_inter = len_inter_x * len_inter_y
    s_all = len1_x * len1_y + len2_x * len2_y
    iou = s_inter / (s_all - s_inter)
    return iou


def __get_len_1Dbox(box1: tf.Tensor, box2: tf.Tensor):
    """
    Shape of each box should be `(..., 2)`
    """
    len1 = tf.abs(box1[..., 0] - box1[..., 1])
    len2 = tf.abs(box2[..., 0] - box2[..., 1])

    x_sorted = tf.sort(tf.concat([box1, box2], axis=-1))
    len_all = tf.abs(x_sorted[..., 0] - x_sorted[..., -1])
    len_inter = tf.maximum(len1 + len2 - len_all, 0.0)

    return len1, len2, len_inter
