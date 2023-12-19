"""
@Author: Conghao Wong
@Date: 2023-08-08 15:52:46
@LastEditors: Conghao Wong
@LastEditTime: 2023-12-06 15:36:55
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as __qpid

from . import original_models
from .__args import PhysicalCircleArgs, SocialCircleArgs
from .ev_sc import EVSCModel, EVSCStructure
from .ev_spc import EVSPCModel, EVSPCStructure
from .msn_sc import MSNSCModel, MSNSCStructure
from .trans_sc import TransformerSCModel, TransformerSCStructure
from .v_sc import VSCModel, VSCStructure
from .v_spc import VSPCModel, VSPCStructure

# Add new args
__qpid.register_new_args(SocialCircleArgs, 'SocialCircle Args')
__qpid.register_new_args(PhysicalCircleArgs, 'PhysicalCircle Args')
__qpid.args.add_arg_alias(['--sc', '-sc', '--socialCircle'],
                          ['--model', 'MKII', '-lb', 'speed', '-la'])

# Register Circle-based models
__qpid.silverballers.register(
    # SocialCircle Models
    evsc=[EVSCStructure, EVSCModel],
    vsc=[VSCStructure, VSCModel],
    msnsc=[MSNSCStructure, MSNSCModel],
    transsc=[TransformerSCStructure, TransformerSCModel],

    # InteractionCircle Models (SocialCircle + PhysicalCircle)
    evspc=[EVSPCStructure, EVSPCModel],
    vspc=[VSPCStructure, VSPCModel],
)

__qpid._log_mod_loaded(__package__)
