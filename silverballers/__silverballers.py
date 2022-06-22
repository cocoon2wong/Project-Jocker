"""
@Author: Conghao Wong
@Date: 2022-06-22 10:36:15
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-22 10:37:16
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from .__baseSilverballers import BaseSilverballers
from .agents import Agent47CModel
from .handlers.__burnwoodC import BurnwoodCModel


class Silverballers47C(BaseSilverballers):

    def __init__(self, terminal_args: list[str]):

        self.set_models(agentModel=Agent47CModel,
                        handlerModel=BurnwoodCModel)

        super().__init__(terminal_args)
