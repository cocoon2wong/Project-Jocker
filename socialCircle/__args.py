"""
@Author: Conghao Wong
@Date: 2023-08-08 15:19:56
@LastEditors: Conghao Wong
@LastEditTime: 2024-01-24 14:44:38
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class SocialCircleArgs(EmptyArgs):

    @property
    def rel_speed(self) -> int:
        """
        Choose whether to use the relative speed or the absolute speed
        as the speed factor in the SocialCircle.
        (Default to the `absolute speed`)
        """
        return self._arg('rel_speed', 0, argtype=STATIC)

    @property
    def Ts(self) -> str:
        """
        The transformation on SocialCircle.
        It could be:
        - `none`: no transformations
        - `fft`: fast Fourier transform
        - `haar`: haar wavelet transform
        - `db2`: DB2 wavelet transform
        """
        return self._arg('Ts', 'none', argtype=STATIC, short_name='Ts')

    @property
    def partitions(self) -> int:
        """
        Partitions in the SocialCircle.
        It should be manually set at each training run.
        """
        return self._arg('partitions', -1, argtype=STATIC)

    @property
    def use_velocity(self) -> int:
        """
        Choose whether to use the velocity factor in the SocialCircle.
        """
        return self._arg('use_velocity', 1, argtype=STATIC)

    @property
    def use_distance(self) -> int:
        """
        Choose whether to use the distance factor in the SocialCircle.
        """
        return self._arg('use_distance', 1, argtype=STATIC)

    @property
    def use_direction(self) -> int:
        """
        Choose whether to use the direction factor in the SocialCircle.
        """
        return self._arg('use_direction', 1, argtype=STATIC)

    @property
    def use_move_direction(self) -> int:
        """
        Choose whether to use the move direction factor in the SocialCircle.
        """
        return self._arg('use_move_direction', 0, argtype=STATIC)
    
    @property
    def use_acc(self) -> int:
        """
        Choose whether to use the acceleration factor in the SocialCircle.
        """
        return self._arg('use_acc', 0, argtype=STATIC)

    def _init_all_args(self):
        super()._init_all_args()

        # Check partitions (`-1` case)
        if ((p := self.partitions) == -1):
            self.log(f'The number of partitions should be set properly. ' +
                     f'Received `{p}`.',
                     level='error', raiseError=ValueError)


class PhysicalCircleArgs(EmptyArgs):

    @property
    def vision_radius(self) -> float:
        """
        The radius of the target agent's vision field when constructing the 
        PhysicalCircle. Radiuses are based on the length that the agent 
        moves during the observation period.
        """
        return self._arg('vision_radius', 2.0, argtype=STATIC)

    @property
    def adaptive_fusion(self) -> int:
        """
        Choose whether to use the adaptive fusion strategy to fuse SocialCircles
        and PhysicalCircles into InteractionCircles.
        """
        return self._arg('adaptive_fusion', 0, argtype=STATIC)

    @property
    def use_empty_seg_maps(self) -> int:
        """
        Choose whether to use empty segmentation maps when computing the
        PhysicalCircle. The empty segmentation map means that EVERYWHERE
        in the scene is available for walking.
        This arg is only used when running ablation studies.
        """
        return self._arg('use_empty_seg_maps', 0, argtype=TEMPORARY)
