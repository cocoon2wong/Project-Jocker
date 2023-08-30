"""
@Author: Conghao Wong
@Date: 2022-09-01 10:38:40
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-30 16:15:33
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

    def __init__(self, anntype: str = None, ref: int = -1,
                 *args, **kwargs):

        super().__init__(anntype, ref,
                         preprocess_input_types=[INPUT_TYPES.OBSERVED_TRAJ],
                         postprocess_input_types=[OUTPUT_TYPES.PREDICTED_TRAJ],
                         *args, **kwargs)

    def update_paras(self, inputs: dict[str, tf.Tensor]) -> None:
        trajs = inputs[INPUT_TYPES.OBSERVED_TRAJ]
        ref = tf.math.mod(self.ref, trajs.shape[-2])
        ref_point = tf.gather(trajs, [ref], axis=-2)
        self.paras = ref_point

    def preprocess(self, inputs: dict[str, tf.Tensor],
                   use_new_paras=True) -> dict[str, tf.Tensor]:
        """
        Move a specific point to (0, 0) according to the reference time step.
        """
        if use_new_paras:
            self.update_paras(inputs)

        ref_point = self.paras
        outputs = {}
        for _type, _input in inputs.items():
            if _input is not None:
                outputs[_type] = self.move(_input, ref_point)

        return outputs

    def postprocess(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        """
        Move trajectories back to their original positions.
        """
        ref_point = self.paras
        outputs = {}
        for _type, _input in inputs.items():
            if _input is not None:
                outputs[_type] = self.move(_input, ref_point, inverse=True)

        return outputs

    def move(self, trajs: tf.Tensor,
             ref_points: tf.Tensor,
             inverse=False):

        ndim = trajs.ndim
        while ref_points.ndim < ndim:
            ref_points = tf.expand_dims(ref_points, axis=-3)

        if inverse:
            ref_points = -1.0 * ref_points

        # start moving
        trajs_moved = trajs - ref_points
        return trajs_moved
