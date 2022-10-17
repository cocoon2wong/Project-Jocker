"""
@Author: Conghao Wong
@Date: 2022-06-20 15:28:14
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-17 11:26:32
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import sys

import codes as C
import silverballers

if __name__ == '__main__':
    args = C.args.Args(terminal_args=sys.argv)

    model = args.model

    if model == 'linear':
        s = C.models.Linear
    
    else:
        s = silverballers.get_structure(model)
    
    t = s(terminal_args=sys.argv)
    t.train_or_test()

    # It is used to debug
    # t.print_info_all()
    