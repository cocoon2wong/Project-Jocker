"""
@Author: Conghao Wong
@Date: 2022-07-27 20:47:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-07-11 10:38:52
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from codes.args import Args
from codes.constant import INTERPOLATION_TYPES

from . import agents, alpha_agents, handlers
from .__args import SilverballersArgs
from .__baseSilverballers import BaseSilverballers
from .__MiniV import MinimalV, MinimalVModel


class SilverballersMKII(BaseSilverballers):

    def __init__(self, terminal_args: list[str]):

        min_args = SilverballersArgs(terminal_args, is_temporary=True)
        a_model_path = min_args.loada
        b_model_path = min_args.loadb

        # Assign the model type of the first-stage subnetwork
        min_args_a = Args(is_temporary=True)._load_from_json(a_model_path)
        agent_model_type = get_model_type(min_args_a.model)

        # Assign the model type of the second-stage subnetwork
        interp_model = INTERPOLATION_TYPES.get_type(b_model_path)
        if interp_model is None:
            min_args_b = Args(is_temporary=True)._load_from_json(b_model_path)
            handler_model_type = get_model_type(min_args_b.model)
        else:
            handler_model_type = get_model_type(interp_model)

        super().__init__(terminal_args,
                         agent_model_type=agent_model_type,
                         handler_model_type=handler_model_type)


__SILVERBALLERS_DICT = dict(

    # Agent Structures and Models

    # MSN
    msna=[agents.MSNAlpha, agents.MSNAlphaModel],
    msnb=[handlers.MSNBeta, handlers.MSNBetaModel],

    # V^2-Net
    va=[agents.VA, agents.VAModel],
    agent=[agents.VA, agents.VAModel],
    vb=[handlers.VB, handlers.VBModel],

    # E-V^2-Net
    eva=[agents.Agent47C, agents.Agent47CModel],

    # agent47 series
    agent47B=[agents.Agent47B, agents.Agent47BModel],
    agent47BE=[agents.Agent47BE, agents.Agent47BEModel],
    agent47BCBE=[agents.Agent47BCBE, agents.Agent47BCBEModel],
    agent47BCE=[agents.Agent47BCE, agents.Agent47BCEModel],
    agent47C=[agents.Agent47C, agents.Agent47CModel],
    agent47CE=[agents.Agent47CE, agents.Agent47CEModel],

    # ALPHA series
    s300g=[alpha_agents.Sieger300Ghost,
           alpha_agents.Sieger300GhostModel],
    alpha=[alpha_agents.AlphaStructure,
           alpha_agents.AlphaModel],
    beta=[alpha_agents.BetaStructure, alpha_agents.BetaModel],

    # Silverballers Structures
    MKII=[SilverballersMKII, None],
    mv=[MinimalV, MinimalVModel],
)

# Interpolation Handlers
__SILVERBALLERS_DICT.update({
    INTERPOLATION_TYPES.LINEAR: [None, handlers.interp.LinearHandlerModel],
    INTERPOLATION_TYPES.LINEAR_SPEED: [None, handlers.interp.LinearSpeedHandlerModel],
    INTERPOLATION_TYPES.LINEAR_ACC: [None, handlers.interp.LinearAccHandlerModel],
    INTERPOLATION_TYPES.NEWTON: [None, handlers.interp.NewtonHandlerModel],
})


def get_structure(model_name: str):
    return __get(model_name)[0]


def get_model_type(model_name: str, experimental=False):
    if experimental:
        index = 2
    else:
        index = 1
    return __get(model_name)[index]


def __get(model_name: str):
    if not model_name in __SILVERBALLERS_DICT.keys():
        raise NotImplementedError(
            f'model type `{model_name}` is not supported.')

    return __SILVERBALLERS_DICT[model_name]
