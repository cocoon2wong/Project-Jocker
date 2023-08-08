"""
@Author: Conghao Wong
@Date: 2023-08-08 15:57:43
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-08 16:42:37
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from ..agents import BaseAgentModel, BaseAgentStructure
from .__args import SocialCircleArgs


class BaseSocialCircleModel(BaseAgentModel):
    def __init__(self, Args: SocialCircleArgs, as_single_model: bool = True,
                 structure=None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        self.args: SocialCircleArgs

    def print_info(self, **kwargs):
        info = {'Transform type (SocialCircle)': self.args.Ts,
                'Partitions in SocialCircle': self.args.partitions,
                'Max partitions in SocialCircle': self.args.obs_frames}

        kwargs.update(**info)
        return super().print_info(**kwargs)


class BaseSocialCircleStructure(BaseAgentStructure):
    ARG_TYPE = SocialCircleArgs
