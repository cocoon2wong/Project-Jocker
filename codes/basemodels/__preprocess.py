"""
@Author: Conghao Wong
@Date: 2022-06-20 16:24:29
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-27 13:52:04
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import tensorflow as tf

from ..utils import ROTATE_BIAS, SCALE_THRESHOLD


def move(trajs: tf.Tensor,
         para_dict: dict[str, tf.Tensor],
         anntype: str,
         ref: int = -1,
         use_new_para_dict=True) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Move a specific point to (0, 0) according to the reference time step.
    Default reference time step is the last obsetvation step.

    :param trajs: observations, shape = `[(batch,) obs, 2]`
    :param ref: reference point, default is `-1`

    :return trajs_moved: moved trajectories
    :return para_dict: a dict of used parameters
    """
    if use_new_para_dict:
        ref_point = trajs[:, ref, :] if len(trajs.shape) == 3\
            else trajs[ref, :]

        # shape is [batch, 1, 2] or [1, 2]
        ref_point = tf.expand_dims(ref_point, -2)
        para_dict['MOVE'] = ref_point

    else:
        ref_point = para_dict['MOVE']

    if len(trajs.shape) == 4:   # (batch, K, n, 2)
        ref_point = ref_point[:, tf.newaxis, :, :]

    trajs_moved = trajs - ref_point

    return trajs_moved, para_dict


def move_back(trajs: tf.Tensor,
              para_dict: dict[str, tf.Tensor],
              anntype: str) -> tf.Tensor:
    """
    Move trajectories back to their original positions.

    :param trajs: trajectories moved to (0, 0) with reference point, shape = `[(batch,) (K,) pred, 2]`
    :param para_dict: a dict of used parameters, which contains `'ref_point': tf.Tensor`

    :return trajs_moved: moved trajectories
    """
    try:
        ref_point = para_dict['MOVE']  # shape = [(batch,) 1, 2]
        if len(ref_point.shape) == len(trajs.shape):
            trajs_moved = trajs + ref_point
        else:   # [(batch,) K, pred, 2]
            trajs_moved = trajs + tf.expand_dims(ref_point, -3)
        return trajs_moved

    except:
        return trajs


def rotate(trajs: tf.Tensor,
           para_dict: dict[str, tf.Tensor],
           anntype: str,
           ref: int = 0,
           use_new_para_dict=True) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Rotate trajectories to the referce angle.

    :param trajs: observations, shape = `[(batch,) obs, 2]`
    :param ref: reference angle, default is `0`

    :return trajs_rotated: moved trajectories
    :return para_dict: a dict of used parameters, `'rotate_angle': tf.Tensor`
    """
    if use_new_para_dict:
        steps = trajs.shape[-2]
        vectors = tf.gather(trajs, steps-1, axis=-2) - \
            tf.gather(trajs, 0, axis=-2)    # (batch, n)

        if anntype == 'coordinate':
            order = [[0, 1]]
        elif anntype == 'boundingbox':
            order = [[0, 1], [2, 3]]
        else:
            raise NotImplementedError(anntype)

        angles = []
        for [x, y] in order:
            vector_x = tf.gather(vectors, x, axis=-1)
            vector_y = tf.gather(vectors, y, axis=-1)
            main_angle = tf.atan((vector_y + ROTATE_BIAS) /
                                 (vector_x + ROTATE_BIAS))
            angle = ref - main_angle
            angles.append(angle)

        para_dict['ROTATE'] = (angles, order)

    else:
        (angles, order) = para_dict['ROTATE']

    trajs_rotated = []
    for angle, [x, y] in zip(angles, order):
        rotate_matrix = tf.stack([[tf.cos(angle), tf.sin(angle)],
                                  [-tf.sin(angle), tf.cos(angle)]])

        if len(rotate_matrix.shape) == 3:
            # reshape to (batch, 2, 2)
            rotate_matrix = tf.transpose(rotate_matrix, [2, 0, 1])

        _trajs = tf.gather(trajs, [x, y], axis=-1)
        _trajs_rotated = _trajs @ rotate_matrix
        trajs_rotated.append(_trajs_rotated)

    trajs_rotated = tf.concat(trajs_rotated, axis=-1)

    return trajs_rotated, para_dict


def rotate_back(trajs: tf.Tensor,
                para_dict: dict[str, tf.Tensor],
                anntype: str) -> tf.Tensor:
    """
    Rotate trajectories back to their original angles.

    :param trajs: trajectories, shape = `[(batch, ) pred, 2]`
    :param para_dict: a dict of used parameters, `'rotate_matrix': tf.Tensor`

    :return trajs_rotated: rotated trajectories
    """
    (angles, order) = para_dict['ROTATE']
    S = tf.cast(trajs.shape, tf.int32)

    trajs_rotated = []
    for angle, [x, y] in zip(angles, order):
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


def scale(trajs: tf.Tensor,
          para_dict: dict[str, tf.Tensor],
          anntype: str,
          ref: float = 1,
          use_new_para_dict=True) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """
    Scale trajectories' direction vector into (x, y), where |x| <= 1, |y| <= 1.
    Reference point when scale is the `last` observation point.

    :param trajs: input trajectories, shape = `[(batch,) obs, 2]`
    :param ref: reference length, default is `1`
    :return trajs_scaled: scaled trajectories
    :return para_dict: a dict of used parameters, contains `scale:tf.Tensor`
    """
    change_flag = False
    if len(trajs.shape) == 2:
        trajs = tf.expand_dims(trajs, 0)    # change into [batch, obs, 2]
        change_flag = True

    if use_new_para_dict:
        steps = trajs.shape[-2]
        vectors = tf.gather(trajs, steps-1, axis=-2) - \
            tf.gather(trajs, 0, axis=-2)    # (batch, n)

        if anntype == 'coordinate':
            order = [[0, 1]]
        elif anntype == 'boundingbox':
            order = [[0, 1], [2, 3]]
        else:
            raise NotImplementedError(anntype)

        scales = []
        for [x, y] in order:
            vector = tf.gather(vectors, [x, y], axis=-1)
            scale = tf.linalg.norm(vector, axis=-1)
            scale = tf.maximum(SCALE_THRESHOLD, scale)

            # reshape into (batch, 1, 1)
            while len(scale.shape) < 3:
                scale = tf.expand_dims(scale, -1)
            scales.append(scale)

        para_dict['SCALE'] = (scales, order)

    else:
        (scales, order) = para_dict['SCALE']

    trajs_scaled = []
    steps = trajs.shape[-2]
    for scale, [x, y] in zip(scales, order):
        _trajs = tf.gather(trajs, [x, y], axis=-1)
        _trajs_end = tf.gather(_trajs, [steps-1], axis=-2)
        _trajs_scaled = (_trajs - _trajs_end) / scale + _trajs_end
        trajs_scaled.append(_trajs_scaled)

    trajs_scaled = tf.concat(trajs_scaled, axis=-1)

    if change_flag:
        trajs_scaled = trajs_scaled[0]

    return trajs_scaled, para_dict


def scale_back(trajs: tf.Tensor,
               para_dict: dict[str, tf.Tensor],
               anntype: str) -> tf.Tensor:
    """
    Scale trajectories back to their original.
    Reference point is the `first` prediction point.

    :param trajs: trajectories, shape = `[(batch,) (K,) pred, 2]`
    :param para_dict: a dict of used parameters, contains `scale:tf.Tensor`
    :return trajs_scaled: scaled trajectories
    """
    original_dim = len(trajs.shape)
    if original_dim < 4:
        for repeat in range(4 - original_dim):
            # change into [batch, K, pred, 2]
            trajs = tf.expand_dims(trajs, -3)

    trajs_scaled = []
    (scales, order) = para_dict['SCALE']
    for scale, [x, y] in zip(scales, order):
        # reshape into (batch, 1, 1, 1)
        while len(scale.shape) < 4:
            scale = tf.expand_dims(scale, -1)

        _trajs = tf.gather(trajs, [x, y], axis=-1)
        _trajs_end = tf.gather(_trajs, [0], axis=-2)
        _trajs_scaled = (_trajs - _trajs_end) * scale + _trajs_end
        trajs_scaled.append(_trajs_scaled)

    trajs_scaled = tf.concat(trajs_scaled, axis=-1)

    if original_dim < 4:
        for repeat in range(4 - original_dim):
            trajs_scaled = trajs_scaled[0]

    return trajs_scaled


def upSampling(trajs: tf.Tensor,
               para_dict: dict[str, tf.Tensor],
               anntype: str,
               sample_time: int,
               use_new_para_dict=True):

    if use_new_para_dict:
        para_dict['UPSAMPLING'] = sample_time
    else:
        sample_time = para_dict['UPSAMPLING']

    original_number = trajs.shape[-2]
    sample_number = sample_time * original_number

    if len(trajs.shape) == 3:   # (batch, n, 2)
        return tf.image.resize(trajs[:, :, :, tf.newaxis], [sample_number, 2])[:, :, :, 0], para_dict

    elif len(trajs.shape) == 4:   # (batch, K, n, 2)
        K = trajs.shape[1]
        results = []
        for k in range(K):
            results.append(tf.image.resize(
                trajs[:, k, :, :, tf.newaxis],
                [sample_number, 2])[:, :, :, 0])

        return tf.transpose(tf.stack(results), [1, 0, 2, 3]), para_dict


def upSampling_back(trajs: tf.Tensor,
                    para_dict: dict[str, tf.Tensor],
                    anntype: str):
    sample_time = para_dict['UPSAMPLING']
    sample_number = trajs.shape[-2]
    original_number = sample_number // sample_time
    original_index = tf.range(original_number) * sample_time

    return tf.gather(trajs, original_index, axis=-2)


def update(new: Union[tuple, list],
           old: Union[tuple, list]) -> tuple:

    if type(old) == list:
        old = tuple(old)
    if type(new) == list:
        new = tuple(new)

    if len(new) < len(old):
        return new + old[len(new):]
    else:
        return new
