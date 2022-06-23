"""
@Author: Conghao Wong
@Date: 2022-06-23 10:16:04
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-23 15:05:21
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from codes.args import BaseArgTable


class VArgs(BaseArgTable):
    def __init__(self, terminal_args: list[str] = None) -> None:
        super().__init__(terminal_args)

    @property
    def key_points(self) -> str:
        """
        A list of key-time-steps to be predicted in the agent model.
        For example, `'0_6_11'`.
        """
        return self._get('key_points', '0_6_11', argtype='static')

    @property
    def Kc(self) -> int:
        """
        Number of hidden categories used in alpha model.
        """
        return self._get('Kc', 20, argtype='static')

    @property
    def points(self) -> int:
        """
        Controls number of points (representative time steps) input to the beta model.
        """
        return self._get('points', 1, argtype='static')

    @property
    def feature_dim(self) -> int:
        """
        (It is unused in this model)
        """
        return self._get('feature_dim', -1, argtype='static')

    @property
    def depth(self) -> int:
        """
        (It is unused in this model)
        """
        return self._get('depth', -1, argtype='static')
