"""
@Author: Conghao Wong
@Date: 2022-09-01 11:15:52
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-30 20:41:38
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...constant import ANN_TYPES, INPUT_TYPES, OUTPUT_TYPES
from ...utils import ROTATE_BIAS, batch_matmul
from .__base import BaseProcessLayer


class Rotate(BaseProcessLayer):
    """
    Rotate trajectories to the reference angle (0.0 degree).
    It also moves all neighbors trajectories (according to the position of
    the target agent at current time step) if they are available.
    """

    def __init__(self, anntype: str, *args, **kwargs):

        super().__init__(anntype,
                         preprocess_input_types=[INPUT_TYPES.OBSERVED_TRAJ,
                                                 INPUT_TYPES.NEIGHBOR_TRAJ],
                         postprocess_input_types=[OUTPUT_TYPES.PREDICTED_TRAJ])

        if anntype != ANN_TYPES.CO_2D:
            raise NotImplementedError(
                f'Rotate is not supported on `{anntype}`.')

        self._angles = None

    @property
    def angles(self) -> tf.Tensor:
        """
        Angles of all oberved trajectories (move vectors).
        Shape is `(batch)`.
        """
        return self._angles

    def update_paras(self, inputs: dict[str, tf.Tensor]) -> None:
        trajs = inputs[INPUT_TYPES.OBSERVED_TRAJ]
        vectors = trajs[..., -1, :] - trajs[..., 0, :]
        main_angles = tf.atan((vectors[..., 1] + ROTATE_BIAS) /
                              (vectors[..., 0] + ROTATE_BIAS))
        angles = 0.0 - main_angles
        self._angles = angles

    def preprocess(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        Rotate trajectories to the reference angle.
        """
        outputs = {}
        for _type, _input in inputs.items():
            if _input is not None:
                outputs[_type] = self.rotate(_input)
        return outputs

    def postprocess(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        Rotate trajectories back to their original angles.
        """
        outputs = {}
        for _type, _input in inputs.items():
            if _input is not None:
                outputs[_type] = self.rotate(_input, inverse=True)
        return outputs

    def rotate(self, trajs: tf.Tensor, inverse=False):
        angles = self.angles
        if inverse:
            angles = -1.0 * angles

        # Rotate matrix: (2, 2, batch) -> (batch, 2, 2)
        rotate_matrix = tf.stack([[tf.cos(angles), tf.sin(angles)],
                                  [-tf.sin(angles), tf.cos(angles)]])
        rotate_matrix = tf.transpose(rotate_matrix, [2, 0, 1])

        while rotate_matrix.ndim < trajs.ndim:
            rotate_matrix = tf.expand_dims(rotate_matrix, -3)

        return batch_matmul(trajs, rotate_matrix)
