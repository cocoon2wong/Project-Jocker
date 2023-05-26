"""
@Author: Conghao Wong
@Date: 2023-05-25 14:51:07
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-26 15:41:50
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Any

import numpy as np

from ....base import BaseManager
from ....constant import INPUT_TYPES
from ....dataset.__splitManager import Clip
from ..__baseInputManager import BaseInputManager
from .__trajMapManager import TrajMapManager


class MapParasManager(BaseInputManager):
    """
    Map Parameters Manager
    ---
    It is used to load trajectory map's parameters from files.
    """

    INPUT_TYPE = INPUT_TYPES.MAP_PARAS

    def __init__(self, manager: BaseManager,
                 name='Map Parameters Manager'):

        super().__init__(manager, name)

    def save(self, *args, **kwargs) -> Any:
        pass

    def load(self, agents: list, *args, **kwargs) -> list:
        # load global map's configs
        config_path = self.temp_file
        config_dict = np.load(config_path, allow_pickle=True).tolist()

        W = config_dict['W']
        b = config_dict['b']

        return np.repeat(np.concatenate([W, b])[np.newaxis],
                         repeats=len(agents), axis=0)

    def get_temp_file_path(self, clip: Clip) -> str:
        map_mgr = self.manager.get_member(TrajMapManager)
        return map_mgr.get_temp_files_paths(clip)['GLOBAL_CONFIG_FILE']
