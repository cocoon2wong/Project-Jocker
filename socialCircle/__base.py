"""
@Author: Conghao Wong
@Date: 2023-08-08 15:57:43
@LastEditors: Conghao Wong
@LastEditTime: 2023-12-28 10:39:50
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from qpid.constant import INPUT_TYPES
from qpid.silverballers import AgentArgs, BaseAgentModel, BaseAgentStructure

from .__args import PhysicalCircleArgs, SocialCircleArgs


class BaseSocialCircleModel(BaseAgentModel):

    include_socialCircle = True
    include_physicalCircle = False

    _s_args: SocialCircleArgs | None = None
    _p_args: PhysicalCircleArgs | None = None

    def __init__(self, Args: AgentArgs, as_single_model: bool = True,
                 structure=None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Init args and input types
        input_types = [INPUT_TYPES.OBSERVED_TRAJ]

        if self.include_socialCircle:
            input_types += [INPUT_TYPES.NEIGHBOR_TRAJ]
            self._s_args = self.args.register_subargs(SocialCircleArgs, 'SC')

        if self.include_physicalCircle:
            input_types += [INPUT_TYPES.SEG_MAP, INPUT_TYPES.SEG_MAP_PARAS]
            self._p_args = self.args.register_subargs(PhysicalCircleArgs, 'PC')

        # Assign input and label types
        self.set_inputs(*input_types)
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

    @property
    def sc_args(self) -> SocialCircleArgs:
        """
        Args for setting up the SocialCircle.
        It may raise `ValueError` if there is no SocialCircle in this model.
        """
        if not self._s_args:
            raise ValueError('SC Args not found in this model!')
        return self._s_args

    @property
    def pc_args(self) -> PhysicalCircleArgs:
        """
        Args for setting up the PhysicalCircle.
        It may raise `ValueError` if there is no PhysicalCircle in this model.
        """
        if not self._p_args:
            raise ValueError('PC Args not found in this model!')
        return self._p_args

    def print_info(self, **kwargs):
        factors = [item for item in ['velocity',
                                     'distance',
                                     'direction',
                                     'move_direction']
                   if getattr(self.sc_args, f'use_{item}')]

        info: dict = {'InteractionCircle Settings': None}

        if self._s_args:
            info.update({
                # 'Transform type (SocialCircle)': self.sc_args.Ts,
                '- The number of circle partitions': self.sc_args.partitions,
                '- Maximum circle partitions': self.args.obs_frames,
                '- Factors in SocialCircle': factors,
            })

        if self._p_args:
            info.update({
                '- Vision radiuses in PhysicalCircle': self.pc_args.vision_radius,
                '- Adaptive Fusion': 'Activated' if self.pc_args.adaptive_fusion
                else 'Disabled',
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
