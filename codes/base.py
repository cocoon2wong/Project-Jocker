"""
@Author: Conghao Wong
@Date: 2022-06-20 16:28:13
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-17 11:38:14
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import logging
from typing import Iterable, TypeVar, Union

import tensorflow as tf
from tqdm import tqdm

from .utils import MAX_PRINT_LIST_LEN
from .args import Args

T = TypeVar('T')


class _BaseManager():
    """
    BaseManager
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

    def __init__(self, name: str = None):
        super().__init__()

        self.name = name

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

    def timebar(self, inputs: T, text='') -> T:
        self.bar = tqdm(inputs, desc=text)
        return self.bar

    def update_timebar(self, item: Union[str, dict], pos='end'):
        """
        Update the tqdm timebar.

        :param item: string or dict to update
        :param pos: position, canbe `'end'` or `'start'`
        """
        if pos == 'end':
            if type(item) is str:
                self.bar.set_postfix_str(item)
            elif type(item) is dict:
                self.bar.set_postfix(item)
            else:
                raise ValueError(item)

        elif pos == 'start':
            self.bar.set_description(item)
        else:
            raise NotImplementedError(pos)

    def print_info(self, **kwargs):
        """
        Print information of the object itself.
        """
        self.print_parameters(**kwargs)

    def print_parameters(self, title='null', **kwargs):
        if title == 'null':
            title = ''

        print(f'\n>>> [{self.name}]: {title}')
        for key, value in kwargs.items():
            if type(value) == tf.Tensor:
                value = value.numpy()

            if (type(value) == list and
                    len(value) > MAX_PRINT_LIST_LEN):
                value = value[:MAX_PRINT_LIST_LEN] + ['...']

            print(f'    - {key}: {value}.')

        print('')

    @staticmethod
    def log_bar(percent, total_length=30):

        bar = (''.join('=' * (int(percent * total_length) - 1))
               + '>')
        return bar


# It is used for type-hinting
class BaseManager(_BaseManager):
    """
    BaseManager
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

    def __init__(self, args: Args = None,
                 manager: _BaseManager = None,
                 name: str = None):

        super().__init__(name)
        self._args: Args = args
        self.manager: _BaseManager = manager
        self.members: list[_BaseManager] = []

        if manager:
            self.manager.members.append(self)

    @property
    def args(self) -> Args:
        if self._args:
            return self._args
        elif self.manager:
            return self.manager.args
        else:
            return None

    @args.setter
    def args(self, value: T) -> T:
        self._args = value

    def get_members_by_type(self, mtype: type[T]) -> list[T]:
        results = []
        for m in self.members:
            if type(m) == mtype:
                results.append(m)

        return results

    def print_info_all(self, include_self=True):
        """
        Print information of the object itself and all its members.
        It is used to debug only.
        """
        if include_self:
            self.print_info(title='DEBUG', object=self, members=self.members)

        for s in self.members:
            s.print_info(title='DEBUG', object=s,
                         manager=self, members=s.members)
            s.print_info_all(include_self=False)

    def print_manager_info(self):
        self.print_parameters(title='Information',
                              name=self.__str__(),
                              type=type(self).__name__,
                              members=self.members,
                              manager=self.manager)


class __SecondaryBar(BaseManager):

    def __init__(self, item: Iterable,
                 manager: BaseManager,
                 desc: str = 'Calculating:',
                 pos: str = 'end',
                 name='Secondary InformationBar Manager'):

        super().__init__(name=name)

        if not '__getitem__' in item.__dir__():
            item = list(item)

        self.item = item
        self.target = manager
        self.desc = desc + ' {}%'
        self.pos = pos

        self.max = len(item)
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= self.max:
            raise StopIteration

        # get value
        value = self.item[self.count]
        self.count += 1

        # update timebar
        percent = (self.count * 100) // self.max
        self.target.update_timebar(item=self.desc.format(percent),
                                   pos=self.pos)

        return value


# It is only used for type-hinting
def SecondaryBar(item: T,
                 manager: BaseManager,
                 desc: str = 'Calculating:',
                 pos: str = 'end',
                 name='Secondary InformationBar Manager') -> T:
    """
    Init

    :param item: an iterable object
    :param manager: target manager object to be updated
    :param desc: text to show on the main timebar
    :param pos: text position, can be `'start'` or `'end'`
    """
    return __SecondaryBar(item, manager, desc, pos, name)
