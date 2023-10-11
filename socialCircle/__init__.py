"""
@Author: Conghao Wong
@Date: 2023-08-08 15:52:46
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-10 16:43:36
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as __qpid

from . import original_models
from .__args import SocialCircleArgs
from .__layers import SocialCircleLayer
from .__sc_EV import EVSCModel, EVSCStructure
from .__sc_MSN import MSNSCModel, MSNSCStructure
from .__sc_transformer import TransformerSCModel, TransformerSCStructure
from .__sc_V import VSCModel, VSCStructure

__qpid.args.add_arg_alias('--sc', ['--model', 'MKII', '-lb', 'speed', '-la'])
__qpid.update_args_dic({SocialCircleArgs: ['SocialCircle Args', 1]})
__qpid.silverballers.register(evsc=[EVSCStructure, EVSCModel],
                              vsc=[VSCStructure, VSCModel],
                              msnsc=[MSNSCStructure, MSNSCModel],
                              transsc=[TransformerSCStructure,
                                       TransformerSCModel])
__qpid._log_mod_loaded(__package__)
