"""
@Author: Conghao Wong
@Date: 2023-05-25 14:51:07
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-29 19:37:40
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Any

import numpy as np

from ....base import BaseManager
from ....constant import INPUT_TYPES
from ....dataset.__splitManager import Clip
from ....utils import (SEG_IMG, WINDOW_EXPAND_METER, WINDOW_EXPAND_PIXEL,
                       WINDOW_SIZE_METER, WINDOW_SIZE_PIXEL)
from ..__baseInputManager import BaseInputManager


class MapParasManager(BaseInputManager):
    """
    Map Parameters Manager
    ---
    It is used to load trajectory map's parameters from files.
    """

    TEMP_FILE = 'configs.npy'
    INPUT_TYPE = INPUT_TYPES.MAP_PARAS

    def __init__(self, manager: BaseManager,
                 name='Map Parameters Manager'):

        super().__init__(manager, name)

        # Parameters
        self.map_type: str = None
        self.a: float = None
        self.e: float = None

        # Variables
        self.__void_map: np.ndarray = None
        self.W: np.ndarray = None
        self.b: np.ndarray = None

    @property
    def void_map(self) -> np.ndarray:
        """
        Get a copy of an empty map.
        """
        return self.__void_map.copy()

    @property
    def use_seg_map(self) -> bool:
        """
        Whether to use segmentation maps instead of
        the calculated trajectory maps.
        """
        if (self.args.use_seg_maps and
            SEG_IMG in self.working_clip.other_files.keys() and
                os.path.exists(self.working_clip.other_files[SEG_IMG])):
            return True
        else:
            return False

    def save(self, trajs: np.ndarray,
             *args, **kwargs) -> Any:

        # Get 2D center points
        t_center = self.C(trajs)

        if t_center.ndim == 3:
            t_center = np.reshape(t_center, [-1, 2])

        x_max = np.max(t_center[:, 0])
        x_min = np.min(t_center[:, 0])
        y_max = np.max(t_center[:, 1])
        y_min = np.min(t_center[:, 1])

        a = self.a
        e = self.e

        self.__void_map = np.zeros([int((x_max - x_min + 2 * e) * a) + 1,
                                    int((y_max - y_min + 2 * e) * a) + 1],
                                   dtype=np.float32)
        self.W = np.array([a, a])
        self.b = np.array([x_min - e, y_min - e])

        np.save(self.temp_file,
                arr=dict(void_map=self.__void_map,
                         W=self.W,
                         b=self.b),)

    def load(self, agents: list, *args, **kwargs) -> list:
        # load global map's configs
        config_path = self.temp_file
        config_dict = np.load(config_path, allow_pickle=True).tolist()

        self.W = config_dict['W']
        self.b = config_dict['b']
        self.__void_map = config_dict['void_map']

        return np.repeat(np.concatenate([self.W, self.b])[np.newaxis],
                         repeats=len(agents), axis=0)

    def init_clip(self, clip: Clip):
        self.map_type = clip.manager.type

        if self.map_type == 'pixel':
            self.a = WINDOW_SIZE_PIXEL
            self.e = WINDOW_EXPAND_PIXEL

        elif self.map_type == 'meter':
            self.a = WINDOW_SIZE_METER
            self.e = WINDOW_EXPAND_METER

        else:
            raise ValueError(self.map_type)

    def real2grid(self, traj: np.ndarray) -> np.ndarray:
        if not type(traj) == np.ndarray:
            traj = np.array(traj)

        grid = ((traj - self.b) * self.W).astype(np.int32)
        return grid

    def C(self, trajs: np.ndarray) -> np.ndarray:
        """
        Get the 2D center point of the input M-dimensional trajectory.
        """
        if trajs.shape[-1] == 2:
            return trajs

        t = self.picker.get_center(trajs)
        if t.shape[-1] > 2:
            t = t[..., :2]
        return t
