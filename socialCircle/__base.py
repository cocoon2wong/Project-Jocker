"""
@Author: Conghao Wong
@Date: 2023-08-08 15:57:43
@LastEditors: Conghao Wong
@LastEditTime: 2023-12-19 16:35:50
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Optional

from qpid.constant import INPUT_TYPES
from qpid.silverballers import AgentArgs, BaseAgentModel, BaseAgentStructure
from qpid.silverballers.__loss import keyl2

from .__args import PhysicalCircleArgs, SocialCircleArgs
from .__loss import SegLoss


class BaseSocialCircleModel(BaseAgentModel):

    include_socialCircle = True
    include_physicalCircle = False

    sc_args: SocialCircleArgs | None = None
    pc_args: PhysicalCircleArgs | None = None

    def __init__(self, Args: AgentArgs, as_single_model: bool = True,
                 structure=None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Init args and input types
        input_types = [INPUT_TYPES.OBSERVED_TRAJ]

        if self.include_socialCircle:
            input_types += [INPUT_TYPES.NEIGHBOR_TRAJ]
            self.sc_args = self.args.register_subargs(
                SocialCircleArgs, 'SCArgs')

        if self.include_physicalCircle:
            input_types += [INPUT_TYPES.SEG_MAP,
                            INPUT_TYPES.SEG_MAP_PARAS]
            self.pc_args = self.args.register_subargs(
                PhysicalCircleArgs, 'PCArgs')

        # Assign input and label types
        self.set_inputs(*input_types)
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

    def print_info(self, **kwargs):
        factors = [item for item in ['velocity',
                                     'distance',
                                     'direction',
                                     'move_direction']
                   if getattr(self.sc_args, f'use_{item}')]

        info: dict = {'InteractionCircle Settings': None}

        if self.sc_args:
            info.update({
                # 'Transform type (SocialCircle)': self.sc_args.Ts,
                '- The number of circle partitions': self.sc_args.partitions,
                '- Maximum circle partitions': self.args.obs_frames,
                '- Factors in SocialCircle': factors,
            })

        if self.pc_args:
            info.update({
                '- Vision radiuses in PhysicalCircle': self.pc_args.vision_radius,
            })

        return super().print_info(**kwargs, **info)


class BaseSocialCircleStructure(BaseAgentStructure):

    MODEL_TYPE: type[BaseSocialCircleModel]

    def __init__(self, terminal_args: list[str] | AgentArgs,
                 manager=None):
        super().__init__(terminal_args, manager)

        if self.args.model_type != 'agent-based':
            self.log('SocialCircle models only support model type `agent-based`.' +
                     f' Current setting is `{self.args.model_type}`.',
                     level='error', raiseError=ValueError)

        # Set losses and metrics
        if self.MODEL_TYPE.include_physicalCircle:
            pc_args = self.args.register_subargs(PhysicalCircleArgs, 'PCArgs')
            if pc_args.use_seg_map_loss:
                self.loss.set({keyl2: 1.0, SegLoss: 5.0})
