"""
@Author: Conghao Wong
@Date: 2021-08-05 15:26:57
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-14 10:23:15
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import re
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from codes.args import Args
from silverballers.__args import SilverballersArgs, AgentArgs, HandlerArgs


FLAG = '<!-- DO NOT CHANGE THIS LINE -->'
TARGET_FILE = './README.md'
MAX_SPACE = 20


def read_comments(args: Args) -> list[str]:

    results = []
    for arg in args._arg_list:

        name = arg
        default = args._args_default[name]
        dtype = type(default).__name__
        argtype = args._arg_type[name]

        doc = getattr(args.__class__, arg).__doc__
        doc = doc.replace('\n', ' ')
        for _ in range(MAX_SPACE):
            doc = doc.replace('  ', ' ')

        s = (f'- `--{name}`: type=`{dtype}`, argtype=`{argtype}`.\n' +
             f' {doc}\n  The default value is `{default}`.')
        results.append(s + '\n')
        print(s)

    return results


def update(md_file, args: list[Args], titles: list[str]):

    new_lines = []
    all_args = []

    for arg, title in zip(args, titles):
        new_lines += [f'\n### {title}\n\n']
        c = read_comments(arg)
        c.sort()

        for new_line in c:
            name = new_line.split('`')[1]
            if name not in all_args:
                all_args.append(name)
                new_lines.append(new_line)

    with open(md_file, 'r') as f:
        lines = f.readlines()
    lines = ''.join(lines)

    try:
        pattern = re.findall(
            f'([\s\S]*)({FLAG})([\s\S]*)({FLAG})([\s\S]*)', lines)[0]
        all_lines = list(pattern[:2]) + new_lines + list(pattern[-2:])

    except:
        flag_line = f'{FLAG}\n'
        all_lines = [lines, flag_line] + new_lines + [flag_line]

    with open(md_file, 'w+') as f:
        f.writelines(all_lines)


if __name__ == '__main__':
    files = [Args(is_temporary=True),
             SilverballersArgs(is_temporary=True),
             AgentArgs(is_temporary=True),
             HandlerArgs(is_temporary=True)]

    titles = ['Basic args',
              'Silverballers args',
              'First-stage silverballers args',
              'Second-stage silverballers args']

    update(TARGET_FILE, files, titles)
