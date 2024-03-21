"""
@Author: Conghao Wong
@Date: 2023-09-06 20:45:28
@LastEditors: Conghao Wong
@LastEditTime: 2024-03-21 09:45:22
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as __qpid

from .__args import VArgs
from .ev import EV, EVModel
from .trans import MinimalV, MinimalVModel
from .v import VA, VB, VAModel, VBModel

__qpid.register_new_args(VArgs, 'V^2-Net Args')
__qpid.silverballers.register(
    # V^2-Net
    va=[VA, VAModel],
    agent=[VA, VAModel],
    vb=[VB, VBModel],

    # E-V^2-Net
    eva=[EV, EVModel],
    agent47C=[EV, EVModel],

    # Other models
    trans=[MinimalV, MinimalVModel],
    mv=[MinimalV, MinimalVModel],
)
