"""
@Author: Conghao Wong
@Date: 2023-06-07 11:08:13
@LastEditors: Conghao Wong
@LastEditTime: 2023-07-17 16:07:02
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from codes.args import DYNAMIC, STATIC, TEMPORARY

from ..base import BaseSilverballersArgs


class AgentArgs(BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

    @property
    def depth(self) -> int:
        """
        Depth of the random noise vector.
        """
        return self._arg('depth', 16, argtype=STATIC)

    @property
    def deterministic(self) -> int:
        """
        Controls if predict trajectories in the deterministic way.
        """
        return self._arg('deterministic', 0, argtype=STATIC)

    @property
    def loss(self) -> str:
        """
        Loss used to train agent models.
        Canbe `'avgkey'` or `'keyl2'` (default).
        """
        return self._arg('loss', 'keyl2', argtype=DYNAMIC)

    @property
    def rel_speed(self) -> int:
        """
        Choose wheather use the relative speed or the absolute speed
        as the speed factor in the SocialCircle (only for Beta Model).
        (Default to the `relative speed`)
        """
        return self._arg('rel_speed', 1, argtype=STATIC)
