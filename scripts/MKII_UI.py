"""
@Author: Conghao Wong
@Date: 2023-07-14 16:42:19
@LastEditors: Conghao Wong
@LastEditTime: 2023-07-14 17:34:03
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os
import sys
import tkinter as tk

from utils import TK_BORDER_WIDTH, TK_TITLE_STYLE, TextboxHandler

DEFAULT_AGENT_MODEL = 'weights/Silverbullet/20230712-102506_beta_new_NEWEST_NEWESTbetazara1'
DEFAULT_HANDLER_MODEL = 'speed'


def run(loada: tk.StringVar,
        loadb: tk.StringVar,
        other_args: tk.StringVar):

    from main import main

    la = loada.get()
    lb = loadb.get()
    oa = other_args.get()

    args = f'main.py --model MKII -la {la} -lb {lb} ' + oa
    main(args.split(' '))


if __name__ == '__main__':

    root = tk.Tk()
    root.title('Silverballers MKII')

    # User input frame
    u_args = {
        # 'background': '#FFFFFF',
        'border': TK_BORDER_WIDTH,
    }

    u_frame = tk.Frame(root, **u_args)
    u_frame.grid(row=1, column=0)

    # Main variables
    loada = tk.StringVar(u_frame, DEFAULT_AGENT_MODEL)
    loadb = tk.StringVar(u_frame, DEFAULT_HANDLER_MODEL)
    other_args = tk.StringVar(u_frame)

    tk.Label(u_frame, text='Silverballers MKII',
             **TK_TITLE_STYLE, **u_args).grid(column=0, row=0, sticky=tk.NW)

    la_args = {
        'anchor': tk.W,
    }
    tk.Label(u_frame, text='Agent Model', **u_args, **la_args).grid(
        column=0, row=1)
    tk.Label(u_frame, text='Handler Model', **u_args, **la_args).grid(
        column=0, row=2)
    tk.Label(u_frame, text='Other Args', **u_args, **la_args).grid(
        column=0, row=3)

    e_args = {
        'width': 36,
    }
    tk.Entry(u_frame, textvariable=loada, **e_args).grid(
        column=1, row=1)
    tk.Entry(u_frame, textvariable=loadb, **e_args).grid(
        column=1, row=2)
    tk.Entry(u_frame, textvariable=other_args, **e_args).grid(
        column=1, row=3)

    tk.Button(u_frame, text='Run Prediction', height=5,
              command=lambda: run(loada, loadb, other_args)).grid(
        column=2, row=1, rowspan=3)

    # Log frame
    l_args = {
        # 'background': '#FFFFFF',
        'border': TK_BORDER_WIDTH,
    }

    log_frame = tk.Frame(root, **l_args)
    log_frame.grid(column=0, row=2)

    logbox = tk.Text(log_frame, width=89, height=20, **l_args)
    (scroll := tk.Scrollbar(log_frame, command=logbox.yview)).pack(
        side=tk.RIGHT, fill=tk.Y)
    logbox.config(yscrollcommand=scroll.set)
    logbox.pack()

    sys.path.insert(0, os.path.abspath('.'))
    import qpid
    qpid.set_log_stream_handler(TextboxHandler(logbox))

    root.mainloop()
