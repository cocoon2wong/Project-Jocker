"""
@Author: Conghao Wong
@Date: 2022-06-20 15:28:14
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-23 15:46:47
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import sys

import codes as C
import silverballers
import vertical

if __name__ == '__main__':
    args = C.args.BaseArgTable(terminal_args=sys.argv)

    model = args.model

    # --------------------
    # Silverballers models
    # --------------------
    if model == 'agent47C':
        s = silverballers.agents.Agent47C

    elif model == 'agent47CE':
        s = silverballers.agents.Agent47CE
    
    elif model == 'sb47C':
        s = silverballers.Silverballers47C

    # ---------------
    # Vertical models
    # ---------------
    elif model in ['va', 'agent']:
        s = vertical.VA

    elif model == 'vb':
        s = vertical.VB

    elif model == 'V':
        s = vertical.V

    elif model == 'mv':
        s = vertical.MinimalV
    
    else:
        raise NotImplementedError(
            'model type `{}` is not supported.'.format(model))

    s(terminal_args=sys.argv).train_or_test()
    