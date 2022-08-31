"""
@Author: Conghao Wong
@Date: 2022-07-27 20:47:50
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-31 10:10:56
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

        self.args = SilverballersArgs(terminal_args)
        minimal_agent_args = Args()._load_from_json(self.args.loada)
        agent_model = get_model(minimal_agent_args.model)

        if not self.args.loadb.startswith('l'):
            minimal_handler_args = Args()._load_from_json(self.args.loadb)
            handler_model = get_model(minimal_handler_args.model)
        else:
            handler_model = None

        self.set_models(agentModel=agent_model, handlerModel=handler_model)

        super().__init__(terminal_args)


__SILVERBALLERS_DICT = dict(

    # Agent Structures and Models

    # V^2-Net
    va=[agents.VA, agents.VAModel],
    agent=[agents.VA, agents.VAModel],

    # agent47 series
    agent47B=[agents.Agent47B, agents.Agent47BModel],
    agent47C=[agents.Agent47C, agents.Agent47CModel],
    agent47CE=[agents.Agent47CE, agents.Agent47CEModel],

    # Handler Structures and Models
    vb=[handlers.VB, handlers.VBModel],

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
