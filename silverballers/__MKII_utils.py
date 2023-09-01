"""
@Author: Conghao Wong
@Date: 2022-07-27 20:47:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-01 11:10:41
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from . import agents, handlers, socialcircle
from .__MiniV import MinimalV, MinimalVModel
from .base import BaseSubnetwork, BaseSubnetworkStructure


class SILVERBALLERS_DICT():
    """
    Silverballers Dictionary
    ---
    A class that manages all silverballers structures and models.

    ### Public Methods

    - Model register:

        ```python
        SILVERBALLERS_DICT.register(**kwargs)
        ```

    - Get model type (class name):

        ```python
        SILVERBALLERS_DICT.get_model(name)
        ```

    - Get model's training structure type (class name):

        ```python
        SILVERBALLERS_DICT.get_structure(name)
        ```
    """

    __S_DICT: dict[str, tuple[type[BaseSubnetworkStructure],
                              type[BaseSubnetwork]]] = {}
    __S_DICT.update(dict(
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

        # SocialCircle models
        transsc=[socialcircle.TransformerSCStructure,
                 socialcircle.TransformerSCModel],
        msnsc=[socialcircle.MSNSCStructure, socialcircle.MSNSCModel],
        vsc=[socialcircle.VSCStructure, socialcircle.VSCModel],
        evsc=[socialcircle.EVSCStructure, socialcircle.EVSCModel],

        # Other models
        mv=[MinimalV, MinimalVModel],
    ))

    # Interpolation Handlers
    __S_DICT.update(handlers.interp.INTERPOLATION_HANDLER_DICT)

    @classmethod
    def register(cls, **kwargs):
        """
        Register new silverballers models.
        Arg format:
        `MODEL_NAME = [STRUCTURE_TYPE, MODEL_TYPE]`

        For example, you can register vertical models by
        ```python
        SILVERBALLERS_DICT.register(va=[VA, VAModel],
                                    vb=[VB, VBModel])
        ```
        """
        for k, v in kwargs.items():
            cls.__S_DICT[k] = v

    @classmethod
    def get_structure(cls, name: str) -> type[BaseSubnetworkStructure]:
        """
        Get structure type of the given model.
        """
        return cls.__get(name)[0]

    @classmethod
    def get_model(cls, name: str) -> type[BaseSubnetwork]:
        """
        Get model type of the given model.
        """
        return cls.__get(name)[1]

    @classmethod
    def __get(cls, name: str):
        if not name in cls.__S_DICT.keys():
            raise NotImplementedError(
                f'Model type `{name}` is not supported.')

        return cls.__S_DICT[name]
