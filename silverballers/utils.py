"""
@Author: Conghao Wong
@Date: 2022-07-27 20:47:50
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-14 09:50:36
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from codes.args import Args

from . import agents, handlers
from .__args import SilverballersArgs
from .__baseSilverballers import BaseSilverballers
from .__silverballers import Silverballers47C, V


class SilverballersMKII(BaseSilverballers):

    def __init__(self, terminal_args: list[str]):

        min_args = SilverballersArgs(terminal_args, is_temporary=True)
        a_model_path = min_args.loada
        b_model_path = min_args.loadb

        # Assign the model type of the first-stage subnetwork
        min_args_a = Args(is_temporary=True)._load_from_json(a_model_path)
        agent_model = get_model(min_args_a.model)

        # Assign the model type of the second-stage subnetwork
        if not b_model_path.startswith('l'):
            min_args_b = Args(is_temporary=True)._load_from_json(b_model_path)
            handler_model = get_model(min_args_b.model)
        else:
            handler_model = None

        self.set_models(agentModel=agent_model, handlerModel=handler_model)

        super().__init__(terminal_args)


__SILVERBALLERS_DICT = dict(

    # Agent Structures and Models

    # MSN
    msna=[agents.MSNAlpha, agents.MSNAlphaModel],
    msnb=[handlers.MSNBeta, handlers.MSNBetaModel],

    # V^2-Net
    va=[agents.VA, agents.VAModel],
    agent=[agents.VA, agents.VAModel],
    vb=[handlers.VB, handlers.VBModel],

    # agent47 series
    agent47B=[agents.Agent47B, agents.Agent47BModel],
    agent47C=[agents.Agent47C, agents.Agent47CModel],
    agent47CE=[agents.Agent47CE, agents.Agent47CEModel],

    # burnwood series
    burnwoodC=[handlers.BurnwoodC, handlers.BurnwoodCModel],
    # burnwoodM=[handlers.BurnwoodM, handlers.BurnwoodMModel],

    # Silverballers Structures (Traditional)
    V=[V, None],
    sb47C=[Silverballers47C, None],
    MKII=[SilverballersMKII, None],
)


def get_structure(model_name: str):
    return __get(model_name)[0]


def get_model(model_name: str):
    return __get(model_name)[1]


def __get(model_name: str):
    if not model_name in __SILVERBALLERS_DICT.keys():
        raise NotImplementedError(
            f'model type `{model_name}` is not supported.')

    return __SILVERBALLERS_DICT[model_name]
