"""
@Author: Conghao Wong
@Date: 2022-09-01 10:38:40
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-31 10:37:45
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...constant import INPUT_TYPES, OUTPUT_TYPES
from .__base import BaseProcessLayer


class Move(BaseProcessLayer):
    """
    Move a specific point to (0, 0) according to the reference time step.
    The default reference time step is the last observation step.
    """

    def __init__(self, anntype: str = None,
                 ref_index: int = None,
                 *args, **kwargs):

        super().__init__(anntype,
                         preprocess_input_types=[INPUT_TYPES.OBSERVED_TRAJ],
                         postprocess_input_types=[OUTPUT_TYPES.PREDICTED_TRAJ])

        self.ref_index = ref_index if ref_index is not None else -1
        self._ref_points = None

    @property
    def ref_points(self) -> tf.Tensor:
        """
        The reference points when moving trajectories.
        Shape is `(batch, 1, dim)`.
        """
        return self._ref_points

    def update_paras(self, inputs: dict[str, tf.Tensor]) -> None:
        trajs = inputs[INPUT_TYPES.OBSERVED_TRAJ]
        self._ref_points = trajs[..., tf.newaxis, self.ref_index, :]

    def preprocess(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        Move a specific point to (0, 0) according to the reference time step.
        """
        outputs = {}
        for _type, _input in inputs.items():
            outputs[_type] = self.move(_input)
        return outputs

    def postprocess(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        Move trajectories back to their original positions.
        """
        outputs = {}
        for _type, _input in inputs.items():
            outputs[_type] = self.move(_input, inverse=True)
        return outputs

    def move(self, trajs: tf.Tensor, inverse=False):
        ref_points = self.ref_points
        if inverse:
            ref_points = -1.0 * ref_points

        while ref_points.ndim < trajs.ndim:
            ref_points = ref_points[..., tf.newaxis, :, :]

        # Start moving
        return trajs - ref_points
