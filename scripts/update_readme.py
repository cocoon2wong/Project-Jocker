"""
@Author: Conghao Wong
@Date: 2021-08-05 15:26:57
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-14 09:40:19
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import qpid
import socialCircle
from qpid.mods import vis

TARGET_FILE = './README.md'


if __name__ == '__main__':
    qpid.help.update_readme(qpid.print_help_info(), TARGET_FILE)
