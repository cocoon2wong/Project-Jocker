"""
@Author: Conghao Wong
@Date: 2022-06-20 15:28:14
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-06 10:20:29
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import sys

import codes as C
import vertical

if __name__ == '__main__':
    args = C.args.BaseArgTable(terminal_args=sys.argv)

    model = args.model

    # ---------------
    # Vertical models
    # ---------------
    if model in ['va', 'agent']:
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
    