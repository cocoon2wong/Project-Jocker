"""
@Author: Conghao Wong
@Date: 2022-06-20 16:28:13
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-03 14:53:29
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import logging
from typing import TypeVar, Union

import tensorflow as tf
from tqdm import tqdm

from .utils import MAX_PRINT_LIST_LEN

T = TypeVar('T')


class BaseObject():
    """
    BaseObject
    ----------
    Base class for all structures.

    Public Methods
    --------------
    ```python
    # log information
    (method) log: (self: BaseObject, s: str, level: str = 'info') -> None

    # print parameters with the format
    (method) print_parameters: (title='null', **kwargs) -> None

    # timebar
    (method) log_timebar: (inputs, text='', return_enumerate=True) -> (enumerate | tqdm)
    ```
    """

    def __init__(self):
        super().__init__()

        # create or restore a logger
        logger = logging.getLogger(name=type(self).__name__)

        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)

            # add file handler
            fhandler = logging.FileHandler(filename='./test.log', mode='a')
            fhandler.setLevel(logging.INFO)

            # add terminal handler
            thandler = logging.StreamHandler()
            thandler.setLevel(logging.INFO)

            # add formatter
            fformatter = logging.Formatter(
                '[%(asctime)s][%(levelname)s] `%(name)s`: %(message)s')
            fhandler.setFormatter(fformatter)

            tformatter = logging.Formatter(
                '[%(levelname)s] `%(name)s`: %(message)s')
            thandler.setFormatter(tformatter)

            logger.addHandler(fhandler)
            logger.addHandler(thandler)

        self.logger = logger
        self.bar: tqdm = None

    def log(self, s: str, level: str = 'info'):
        """
        Log infomation to files and console

        :param s: text to log
        :param level: log level, canbe `'info'` or `'error'` or `'debug'`
        """
        if level == 'info':
            self.logger.info(s)

        elif level == 'error':
            self.logger.error(s)

        elif level == 'debug':
            self.logger.debug(s)

        else:
            raise NotImplementedError

        return s

    @staticmethod
    def timebar(inputs: T, text='') -> T:
        return tqdm(inputs, desc=text)

    @staticmethod
    def update_timebar(timebar: tqdm, item: Union[str, dict], pos='end'):
        if timebar is None:
            return timebar

        if pos == 'end':
            if type(item) is str:
                timebar.set_postfix_str(item)
            elif type(item) is dict:
                timebar.set_postfix(item)
            else:
                raise ValueError(item)

        elif pos == 'start':
            timebar.set_description(item)
        else:
            raise NotImplementedError(pos)

        return timebar

    @staticmethod
    def print_parameters(title='null', **kwargs):
        print('\n>>> ' + title + ':')
        for key, value in kwargs.items():

            if type(value) == tf.Tensor:
                value = value.numpy()

            if (type(value) == list and
                len(value) > MAX_PRINT_LIST_LEN):
                value = value[:MAX_PRINT_LIST_LEN] + ['...']

            print('    - {}: {}.'.format(key, value))

        print('')

    @staticmethod
    def log_bar(percent, total_length=30):

        bar = (''.join('=' * (int(percent * total_length) - 1))
               + '>')
        return bar
