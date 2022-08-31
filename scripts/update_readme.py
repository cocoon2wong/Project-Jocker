"""
@Author: Conghao Wong
@Date: 2021-08-05 15:26:57
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-31 10:09:36
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import re

FLAG = '<!-- DO NOT CHANGE THIS LINE -->'
TARGET_FILE = './README.md'
MAX_SPACE = 20


def read_comments(file) -> list[str]:
    with open(file, 'r') as f:
        lines = f.readlines()

    lines = ''.join(lines)
    args = re.findall('@property[^@]*', lines)

    results = []
    for arg in args:
        name = re.findall('(def )(.+)(\()', arg)[0][1]
        dtype = re.findall('(-> )(.*)(:)', arg)[0][1]
        argtype = re.findall('(argtype=)(.*)(\))', arg)[0][1]
        default = re.findall('(, )(.*)(, arg)', arg)[0][1]
        comments = re.findall('(""")([\S\s]+)(""")', arg)[0][1]
        comments = comments.replace('\n', ' ')
        for _ in range(MAX_SPACE):
            comments = comments.replace('  ', ' ')

        comments = re.findall('( *)(.*)( *)', comments)[0][1]

        if comments.endswith('. '):
            comments = comments[:-1]

        s = (f'- `--{name}`, type=`{dtype}`, argtype=`{argtype}`.\n  ' +
             f'{comments}\n  The default value is `{default}`.')
        results.append(s + '\n')
        print(s)

    return results


def update(md_file, files: list[str], titles: list[str]):

    new_lines = []
    for f, title in zip(files, titles):
        new_lines += [f'\n### {title}\n\n']
        c = read_comments(f)
        c.sort()
        new_lines += c

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
    for model in ['Silverballers']:
        files = ['./codes/args/__args.py',
                 f'./{model}/__args.py']
        titles = ['Basic args',
                  f'{model} args']
        update(TARGET_FILE.format(model), files, titles)
