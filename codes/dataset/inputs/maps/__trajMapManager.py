"""
@Author: Conghao Wong
@Date: 2023-05-22 16:26:35
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-22 20:36:19
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Any

import cv2
import numpy as np

from ....base import SecondaryBar
from ....constant import INPUT_TYPES
from ....utils import (MAP_HALF_SIZE, WINDOW_EXPAND_METER, WINDOW_EXPAND_PIXEL,
                       WINDOW_SIZE_METER, WINDOW_SIZE_PIXEL)
from ...__splitManager import Clip
from ...trajectories import Agent
from ..__baseInputManager import BaseInputManager, BaseManager
from .__utils import add, cut, pooling2D


class TrajMapManager(BaseInputManager):
    """
    Trajectory Map Manager
    ---
    The trajectory map is a map that builds from all agents'
    observed trajectories. It indicates all possible walkable
    areas around the target agent. The value of the trajectory map
    is in the range `[0, 1]`. A higher value indicates that
    the area may not walkable.
    """

    TEMP_FILES = {'FILE': 'trajMap.npy',
                  'CONFIG_FILE': 'trajMap_configs.npy',
                  'GLOBAL_FILE': 'trajMap.png',
                  'GLOBAL_CONFIG_FILE': 'trajMap_configs.npy'}

    MAP_NAME = 'Trajectory Map'
    INPUT_TYPE = INPUT_TYPES.MAP

    def __init__(self, manager: BaseManager,
                 pool_maps=False,
                 name='Trajectory Map Manager'):

        super().__init__(manager, name)

        self.POOL = pool_maps

        # Parameters
        self.map_type: str = None
        self.a: float = None
        self.e: float = None

        # Configs
        self.HALF_SIZE = MAP_HALF_SIZE
        self.void_map: np.ndarray = None
        self.W: np.ndarray = None
        self.b: np.ndarray = None
        self.map: np.ndarray = None

        if pool_maps:
            self.TEMP_FILES['FILE_WITH_POOLING'] = 'trajMap_pooling.npy'

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

    def run(self, clip: Clip,
            trajs: np.ndarray,
            agents: list[Agent],
            *args, **kwargs) -> list:

        return super().run(clip, trajs, agents)

    def save(self, trajs: np.ndarray,
             agents: list[Agent],
             *args, **kwargs) -> Any:

        # Build and save global trajectory map
        self.map = self.build_global_map(trajs)
        self.build_local_maps(agents)

    def load(self, *args, **kwargs) -> list:
        # load global map's configs
        config_path = self.temp_files['GLOBAL_CONFIG_FILE']
        config_dict = np.load(config_path, allow_pickle=True).tolist()

        self.void_map = config_dict['void_map']
        self.W = config_dict['W']
        self.b = config_dict['b']

        # Load maps from the saved file
        if not self.POOL:
            f = self.temp_files['FILE']
        else:
            f = self.temp_files['FILE_WITH_POOLING']

        return 0.5 * np.load(f, allow_pickle=True)

    def init_global_map(self, init_trajs: np.ndarray):
        """
        Init the trajectory map via a list of agents.
        Shape of the `init_trajs` should be `((batch), obs, 2)`.
        """
        if len(init_trajs.shape) == 3:
            init_trajs = np.reshape(init_trajs, [-1, 2])

        x_max = np.max(init_trajs[:, 0])
        x_min = np.min(init_trajs[:, 0])
        y_max = np.max(init_trajs[:, 1])
        y_min = np.min(init_trajs[:, 1])

        a = self.a
        e = self.e

        self.void_map = np.zeros([int((x_max - x_min + 2 * e) * a) + 1,
                                 int((y_max - y_min + 2 * e) * a) + 1],
                                 dtype=np.float32)
        self.W = np.array([a, a])
        self.b = np.array([x_min - e, y_min - e])

    def build_global_map(self, trajs: np.ndarray,
                         source: np.ndarray = None):
        """
        Build and save the global trajectory map.
        - Saved files: `GLOBAL_FILE`, `GLOBAL_CONFIG_FILE`.
        """
        if source is None:
            if self.void_map is None:
                self.init_global_map(trajs)

            source = self.void_map

        # build the global trajectory map
        source = source.copy()
        source = add(source,
                     self.real2grid(trajs),
                     amplitude=[1],
                     radius=7)

        source = np.minimum(source, 30)
        source = 1 - source / np.max(source)

        # save global trajectory map
        cv2.imwrite(self.temp_files['GLOBAL_FILE'], 255 * source)

        # save global map's configs
        np.save(self.temp_files['GLOBAL_CONFIG_FILE'],
                arr=dict(void_map=self.void_map,
                         W=self.W,
                         b=self.b),)

        return source

    def build_local_maps(self, agents: list[Agent],
                         source: np.ndarray = None):
        """
        Build and save local trajectory maps for all agents.

        - Required files: `self.map` (`GLOBAL_FILE` and `GLOBAL_CONFIG_FILE`).
        - Saved file: `FILE`.
        """
        maps = []
        for agent in SecondaryBar(agents,
                                  manager=self.manager,
                                  desc=f'Building {self.MAP_NAME}...'):

            # Cut the local trajectory map from the global map
            # Center point: the last observed point
            center_real = agent.traj[-1:, :]
            center_pixel = self.real2grid(center_real)
            local_map = cut(self.map, center_pixel, self.HALF_SIZE)[0]
            maps.append(local_map)

        # Save maps
        np.save(self.temp_files['FILE'], maps)

        # Pool the maps
        maps_pooling = pooling2D(np.array(maps))
        np.save(self.temp_files['FILE_WITH_POOLING'], maps_pooling)

    def real2grid(self, traj: np.ndarray) -> np.ndarray:
        if not type(traj) == np.ndarray:
            traj = np.array(traj)

        grid = ((traj - self.b) * self.W).astype(np.int32)
        return grid
