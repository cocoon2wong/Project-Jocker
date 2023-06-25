"""
@Author: Conghao Wong
@Date: 2022-06-20 21:40:55
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-25 10:39:04
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from codes.constant import INPUT_TYPES
from codes.managers import Structure
from codes.training import loss

from ..base import BaseSubnetwork, BaseSubnetworkStructure
from ..loss import avgKey, keyl2
from .__args import AgentArgs


class BaseAgentModel(BaseSubnetwork):

    def __init__(self, Args: AgentArgs,
                 as_single_model: bool = True,
                 structure: Structure = None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Type hinting
        self.args: AgentArgs
        self.structure: BaseAgentStructure

        # Model input types
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ)


class BaseAgentStructure(BaseSubnetworkStructure):

    SUBNETWORK_INDEX = '1'
    ARG_TYPE = AgentArgs
    MODEL_TYPE: type[BaseAgentModel] = None

    def __init__(self, terminal_args: list[str],
                 manager: Structure = None,
                 as_single_model: bool = True):

        super().__init__(terminal_args, manager, as_single_model)

        # For type hinting
        self.args: AgentArgs
        self.model: BaseAgentModel

        # Configs
        if self.args.deterministic:
            self.args._set('Kc', 1)
            self.args._set('K_train', 1)
            self.args._set('K', 1)

        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Losses and metrics
        if self.args.loss == 'keyl2':
            self.loss.set({keyl2: 1.0})
        elif self.args.loss == 'avgkey':
            self.loss.set({avgKey: 1.0})
        else:
            raise ValueError(self.args.loss)

        self.metrics.set({avgKey: 1.0,
                          loss.FDE: 0.0})

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        s = f'python main.py --model MKII --loada {self.args.load} --loadb l'
        self.log(f'You can run `{s}` to start the silverballers evaluation.')
