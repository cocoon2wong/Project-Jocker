"""
@Author: Conghao Wong
@Date: 2023-08-08 15:19:56
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-17 10:13:40
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from codes.args import DYNAMIC, STATIC, TEMPORARY

from ..agents import AgentArgs


class SocialCircleArgs(AgentArgs):

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
        Set it to `-1` to adapt to different observation/prediction
        length settings.
        """
        return self._arg('partitions', -1, argtype=STATIC)

    def _init_all_args(self):
        super()._init_all_args()

        # Set partitions for `-1` case
        if self.partitions == -1:
            self._set('partitions', self.obs_frames,
                      verbose=self._verbose_mode)
