"""
@Author: Conghao Wong
@Date: 2022-06-23 10:45:52
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-23 15:14:06
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from silverballers import BaseSilverballers

from .__Valpha import VAModel
from .__Vbeta import VBModel


class V(BaseSilverballers):

    """
    Vertical model.

    - keypoints-interpolation two subnetworks;
    - implements on agents' trajectory spectrums.
    """

    def __init__(self, terminal_args: list[str]):

        self.set_models(agentModel=VAModel,
                        handlerModel=VBModel)
                        
        super().__init__(terminal_args)
