"""
@Author: Conghao Wong
@Date: 2022-11-11 12:41:16
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-14 11:06:23
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import json
import os
from typing import Any

from ..utils import dir_check
from .__baseObject import BaseObject

STATIC = 'static'
DYNAMIC = 'dynamic'
TEMPORARY = 'temporary'


class ArgsManager(BaseObject):
    """
    Args Manager
    ---
    """

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:
        """
        :param terminal_args: A set of args that received from the user input.
        :param is_temporary: Controls whether this `Args` object is a set of\
            temporary args or a set of args used for training. Temporary args\
            will not initialize the `log_dir`.
        """
        super().__init__(name='Args Manager')

        self._is_temporary = is_temporary
        self._init_done = False

        # Args that load from the saved JSON file
        self._args_load: dict[str, Any] = {}

        # Args obtained from terminal
        self._args_runnning: dict[str, Any] = {}

        # Args that are set manually
        self._args_manually: dict[str, Any] = {}

        # The default args (manually)
        self._args_default_manually: dict[str, Any] = {}

        # The default args (default)
        self._args_default: dict[str, Any] = {}

        self._arg_list = []
        # Types of all args, e.g. `STATIC`.
        self._arg_type: dict[str, str] = {}

        # Short names of all args.
        # Keys: short names.
        # Values: Full names.
        self._arg_short_name: dict[str, str] = {}

        # Register all args used in this object
        self._visit_args()
        self._init_done = True

        # Load args and run preprocess methods
        if terminal_args:
            self._load_from_terminal(terminal_args)

        if (l := self.load) != 'null':
            self._load_from_json(l)

        self._visit_args()

    def _visit_args(self):
        """
        Vist all args.
        """
        for arg in self.__dir__():
            if not arg.startswith('_'):
                getattr(self, arg)

    def _load_from_json(self, dir_path: str):
        """
        Load args from the saved JSON file.
        """
        try:
            arg_paths = [(p := os.path.join(dir_path, item)) for item in os.listdir(dir_path) if (
                item.endswith('args.json'))]

            with open(p, 'r') as f:
                json_dict = json.load(f)

            self._args_load = json_dict

        except:
            raise ValueError(f'Failed to load args from `{dir_path}`.')

        return self

    def _load_from_terminal(self, argv: list[str]):
        """
        Load args from the user inputs.
        """
        dic = {}

        index = 1
        while True:
            try:
                if argv[index].startswith('--'):
                    name = argv[index][2:]
                    value = argv[index+1]

                elif argv[index].startswith('-'):
                    name = argv[index][1:]
                    name = self._arg_short_name[name]
                    value = argv[index+1]

                else:
                    index += 1
                    continue

                dic[name] = value
                index += 2

            except IndexError:
                break

            except KeyError:
                if self._is_temporary:
                    index += 2
                else:
                    self.log(f'The abbreviation `-{name}` was not found,' +
                             ' please check your spelling.',
                             level='error', raiseError=KeyError)

        self._args_runnning = dic
        return self

    def _save_as_json(self, target_dir: str):
        """
        Save current args into a JSON file.
        """
        dir_check(target_dir)
        json_path = os.path.join(target_dir, 'args.json')

        names = [n for (n, v) in self._arg_type.items() if v != TEMPORARY]
        names.sort()
        values = [getattr(self, n) for n in names]

        with open(json_path, 'w+') as f:
            json.dump(dict(zip(names, values)), f,
                      separators=(',\n', ':'))

    def _get_args_by_index_and_name(self, index: int, name: str):
        if index == 0:
            dic = self._args_load
        elif index == 1:
            dic = self._args_runnning
        elif index == 99:
            dic = self._args_manually
        elif index == -1:
            dic = self._args_default_manually
        else:
            raise ValueError('Args index not exist.')

        return dic[name] if name in dic.keys() else None

    def _set(self, name: str, value: Any):
        """
        Set argument manually.
        """
        self._args_manually[name] = value

    def _set_default(self, name: str, value: Any):
        """
        Set default argument values.
        """
        self._args_default_manually[name] = value

    def _arg(self, name: str,
             default: Any,
             argtype: str,
             short_name: str = None,
             preprocess=None):

        if not self._init_done:
            self._register(name, default, argtype, short_name)
            return None

        else:
            if preprocess is not None:
                preprocess(self)
            return self._get(name)

    def _register(self, name: str,
                  default: any,
                  argtype: str,
                  short_name: str = None):
        """
        Register a new arg.
        """
        if not name in self._arg_list:
            self._arg_list.append(name)
            self._arg_type[name] = argtype
            self._args_default[name] = default

            if short_name:
                self._arg_short_name[short_name] = name

    def _get(self, name: str):
        """
        Get value of a arg.

        :param name: name of the arg
        :param default: default value of the arg
        :param argtype: type of the arg, can be
            - `STATIC`
            - `DYNAMIC`
            - `TEMPORARY`
            - ...
        """

        # arg dict index:
        # _args_load: 0
        # _args_running: 1
        # _args_manually: 99
        # _args_default: -1

        argtype = self._arg_type[name]
        default = self._args_default[name]

        if argtype == STATIC:
            order = [99, 0, 1, -1]
        elif argtype == DYNAMIC:
            order = [99, 1, 0, -1]
        elif argtype == TEMPORARY:
            order = [99, 1, -1]
        else:
            raise ValueError('Wrong arg type.')

        # Get args from all dictionaries.
        value = None
        for index in order:
            value = self._get_args_by_index_and_name(index, name)

            if value is not None:
                break
            else:
                continue

        if value is None:
            value = default

        value = type(default)(value)

        return value
