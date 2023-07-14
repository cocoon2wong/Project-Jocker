"""
@Author: Conghao Wong
@Date: 2022-06-20 15:30:24
@LastEditors: Conghao Wong
@LastEditTime: 2023-07-14 16:35:17
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from . import args, base, basemodels, managers, models, training, utils


def set_log_path(p: str):
    """
    Set the path of the log file.
    """
    utils.LOG_FILE = p


def set_log_stream_handler(handler):
    """
    Set the log handler (which handles terminal-like outputs).
    Type of the handler should be `logging.Handler`.
    """
    utils.LOG_STREAM_HANDLER = handler
