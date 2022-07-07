"""
@Author: Conghao Wong
@Date: 2022-06-20 15:28:14
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-07 21:46:38
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
    
    elif model == 'agent47CNew':
        s = silverballers.agents.Agent47CNew

    elif model == 'agent47CE':
        s = silverballers.agents.Agent47CE

    elif model == 'agent47CExp':
        s = silverballers.agents.Agent47CExperimental
    
    elif model == 'sb47C':
        s = silverballers.Silverballers47C
    
    elif model == 'sb47CExp':
        s = silverballers.Silverballers47CExperimental

    elif model == 'sb47CNew':
        s = silverballers.Silverballers47CNew
    
    else:
        raise NotImplementedError(
            'model type `{}` is not supported.'.format(model))

    s(terminal_args=sys.argv).train_or_test()
    