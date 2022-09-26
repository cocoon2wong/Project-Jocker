"""
@Author: Conghao Wong
@Date: 2022-06-21 15:53:48
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-26 15:30:40
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import cv2
import numpy as np

from ..base import BaseManager
from ..utils import (AVOID_SIZE, INTEREST_SIZE, WINDOW_EXPAND_METER,
                     WINDOW_EXPAND_PIXEL, WINDOW_SIZE_METER, WINDOW_SIZE_PIXEL)
from .__agent import Agent

MASK = cv2.imread('./figures/mask_circle.png')[:, :, 0]/50
MASKS = {}

DECAY_P = np.array([[0.0, 0.7, 1.0], [1.0, 1.0, 0.5]])
DECAYS = {}


class MapManager(BaseManager):
    """
    Map Manager
    -----------
    Map manager that manages agent's context map.

    Usage
    -----
    ```python
    # build guidanceMap
    >>> MapManager.build_guidance_map(
            self:MapManager,
            agents:list[Agent],
            source=None,
            regulation=True
        ) -> np.ndarray

    # build socialMap (Attention: return `self`)
    >>> MapManager.build_social_map(
            self:MapManager,
            target_agent:Agent,
            traj_neighbors=[],
            source=None,
            regulation=True
        ) -> MapManager    
    ```
    """

    def __init__(self, manager: BaseManager,
                 map_type: str,
                 init_trajs: np.ndarray = None,
                 init_manager=None):

        super().__init__(manager.args, manager)

        self.map_type = map_type

        if init_manager:
            self.void_map, self.W, self.b = [init_manager.void_map,
                                             init_manager.W,
                                             init_manager.b]
        else:
            self.void_map, self.W, self.b = self.init_guidance_map(init_trajs)

    @property
    def real2grid_paras(self) -> np.ndarray:
        """
        a set of parameters that transfer real positions to the grid positions.
        shape = (2, 2), where the first line is the `W`, and the second is the `b`.
        Formally, `xg = xr * W[0] + b[0], yg = yr * W[1] + b[1]`.
        """
        return np.stack([self.W, self.b])   # (2, 2)

    def init_guidance_map(self, init_trajs: np.ndarray):
        """
        Init the trajectory map via a list of agents.

        :param init_trajs: trajectories to init the guidance map.
            shape should be `((batch), obs, 2)`

        :return guidance_map: initialized trajectory map
        :return W: map parameter `W`
        :return b: map parameter `b`
        """

        traj = init_trajs

        # shape of `traj` should be [*, *, 2] or [*, 2]
        if len(traj.shape) == 3:
            traj = np.reshape(traj, [-1, 2])

        x_max = np.max(traj[:, 0])
        x_min = np.min(traj[:, 0])
        y_max = np.max(traj[:, 1])
        y_min = np.min(traj[:, 1])

        if self.map_type == 'pixel':
            a = WINDOW_SIZE_PIXEL
            e = WINDOW_EXPAND_PIXEL

        elif self.map_type == 'meter':
            a = WINDOW_SIZE_METER
            e = WINDOW_EXPAND_METER

        else:
            raise ValueError(self.map_type)

        guidance_map = np.zeros([int((x_max - x_min + 2 * e) * a) + 1,
                                 int((y_max - y_min + 2 * e) * a) + 1])
        W = np.array([a, a])
        b = np.array([x_min - e, y_min - e])

        return guidance_map.astype(np.float32), W, b

    def build_guidance_map(self, trajs: np.ndarray,
                           source: np.ndarray = None,
                           save: str = None) -> np.ndarray:
        """
        Build guidance map

        :param agents: a list of agents or trajectories to calculate the map
        :param source: source map, default are zeros
        :param save: path for saving the guidance map. Support `.jpg` or `.png` format.
        """

        if source is None:
            source = self.void_map

        source = source.copy()
        source = self._add(source,
                           self.real2grid(trajs),
                           amplitude=[1],
                           radius=7,
                           add_mask=MASK,
                           max_limit=False)

        source = np.minimum(source, 30)
        source = 1 - source / np.max(source)

        if save:
            cv2.imwrite(save, 255 * source)

        return source

    def build_social_map(self, target_agent: Agent,
                         source: np.ndarray = None,
                         regulation=True,
                         max_neighbor=15) -> np.ndarray:
        """
        Build social map
        TODO: Social maps for M-dimensional trajectories

        :param target_agent: target `Agent` object to calculate the map
        :param source: source map, default are zeros
        :param regulation: controls if scale the map into [0, 1]
        """

        if type(source) == type(None):
            source = self.void_map

        source = source.copy()

        # Destination
        source = self._add(target_map=source,
                           grid_trajs=self.real2grid(target_agent.pred_linear),
                           amplitude=[-2],
                           radius=INTEREST_SIZE,
                           add_mask=MASK,
                           max_limit=False)

        # Interplay
        traj_neighbors = target_agent.pred_linear_neighbor
        amp_neighbors = []

        vec_target = target_agent.pred_linear[-1] - target_agent.pred_linear[0]
        len_target = calculate_length(vec_target)

        vec_neighbor = traj_neighbors[:, -1] - traj_neighbors[:, 0]

        if len_target >= 0.05:
            cosine = activation(
                calculate_cosine(vec_target[np.newaxis, :], vec_neighbor),
                a=1.0,
                b=0.2)
            velocity = (calculate_length(vec_neighbor) /
                        calculate_length(vec_target[np.newaxis, :]))

        else:
            cosine = np.ones(len(traj_neighbors))
            velocity = 2

        amp_neighbors = - cosine * velocity

        amps = amp_neighbors.tolist()
        trajs = traj_neighbors.tolist()

        if len(trajs) > max_neighbor + 1:
            trajs = np.array(trajs)
            dis = calculate_length(trajs[:1, 0, :] - trajs[:, 0, :])
            index = np.argsort(dis)
            trajs = trajs[index[:max_neighbor+1]]

        source = self._add(target_map=source,
                           grid_trajs=self.real2grid(trajs),
                           amplitude=amps,
                           radius=AVOID_SIZE,
                           add_mask=MASK,
                           max_limit=False)

        if regulation:
            if (np.max(source) - np.min(source)) <= 0.01:
                source = 0.5 * np.ones_like(source)
            else:
                source = (source - np.min(source)) / \
                    (np.max(source) - np.min(source))

        return source

    @staticmethod
    def cut_map(maps: np.ndarray,
                centers: np.ndarray,
                half_size: int) -> np.ndarray:
        """
        Cut original maps into small local maps

        :param maps: maps, shape = (batch, a, b)
        :param centers: center positions (in grids), shape = (batch, 2)
        """
        batch, a, b = maps.shape[-3:]
        centers = centers.astype(np.int32)

        centers = np.maximum(centers, half_size)
        centers = np.array([np.minimum(centers[:, 0], a - half_size),
                            np.minimum(centers[:, 1], b - half_size)]).T

        cuts = []
        for m, c in zip(maps, centers):
            cuts.append(m[c[0] - half_size: c[0] + half_size,
                          c[1] - half_size: c[1] + half_size])

        return np.array(cuts)

    def _add(self, target_map: np.ndarray,
             grid_trajs: np.ndarray,
             amplitude: np.ndarray,
             radius: int,
             add_mask,
             max_limit=False):

        if len(grid_trajs.shape) == 2:
            grid_trajs = grid_trajs[np.newaxis, :, :]

        n_traj, traj_len, dim = grid_trajs.shape[:3]

        if not traj_len in DECAYS.keys():
            DECAYS[traj_len] = np.interp(np.linspace(0, 1, traj_len),
                                         DECAY_P[0],
                                         DECAY_P[1])

        if not radius in MASKS.keys():
            MASKS[radius] = cv2.resize(add_mask, (radius*2+1, radius*2+1))

        a = np.array(amplitude)[:, np.newaxis] * \
            DECAYS[traj_len] * \
            np.ones([n_traj, traj_len], dtype=np.int32)

        points = np.reshape(grid_trajs, [-1, dim])
        amps = np.reshape(a, [-1])

        target_map = target_map.copy()
        target_map = self._add_one_traj(target_map,
                                        points, amps, radius,
                                        MASKS[radius],
                                        max_limit=max_limit)

        return target_map

    def real2grid(self, traj: np.ndarray) -> np.ndarray:
        if not type(traj) == np.ndarray:
            traj = np.array(traj)

        grid = ((traj - self.b) * self.W).astype(np.int32)
        return grid

    def _add_one_traj(self, source_map: np.ndarray,
                      traj: np.ndarray,
                      amplitude: float,
                      radius: int,
                      add_mask: np.ndarray,
                      max_limit=False):

        new_map = np.zeros_like(source_map)
        for pos, a in zip(traj, amplitude):
            if (pos[0]-radius >= 0
                and pos[1]-radius >= 0
                and pos[0]+radius+1 < new_map.shape[0]
                    and pos[1]+radius+1 < new_map.shape[1]):

                new_map[pos[0]-radius:pos[0]+radius+1,
                        pos[1]-radius:pos[1]+radius+1] = \
                    a * add_mask + new_map[pos[0]-radius:pos[0]+radius+1,
                                           pos[1]-radius:pos[1]+radius+1]

        if max_limit:
            new_map = np.sign(new_map)

        return new_map + source_map


def calculate_cosine(vec1: np.ndarray,
                     vec2: np.ndarray):

    length1 = np.linalg.norm(vec1, axis=-1)
    length2 = np.linalg.norm(vec2, axis=-1)

    return (np.sum(vec1 * vec2, axis=-1) + 0.0001) / ((length1 * length2) + 0.0001)


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)


def activation(x: np.ndarray, a=1, b=1):
    return np.less_equal(x, 0) * a * x + np.greater(x, 0) * b * x
