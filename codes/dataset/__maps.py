"""
@Author: Conghao Wong
@Date: 2022-06-21 15:53:48
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-09 18:36:53
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import numpy as np
import tensorflow as tf

from ..base import BaseManager, SecondaryBar
from ..utils import (AVOID_SIZE, INTEREST_SIZE, MAP_HALF_SIZE,
                     WINDOW_EXPAND_METER, WINDOW_EXPAND_PIXEL,
                     WINDOW_SIZE_METER, WINDOW_SIZE_PIXEL, dir_check)
from .__agent import Agent

MASK = cv2.imread('./figures/mask_circle.png')[:, :, 0]/50
MASKS = {}

DECAY_P = np.array([[0.0, 0.7, 1.0], [1.0, 1.0, 0.5]])
DECAYS = {}


POOLING_LAYER = tf.keras.layers.MaxPool2D([5, 5])


class __BaseMapManager(BaseManager):

    def __init__(self, manager: BaseManager,
                 map_type: str,
                 base_path: str,
                 name='Interaction Maps Manager'):

        super().__init__(manager=manager, name=name)

        self.map_type = map_type
        self.dir = dir_check(base_path)
        self.path = os.path.join(base_path, '{}')

        self.FILE = None
        self.FILE_WITH_POOLING = None
        self.CONFIG_FILE = None

        self.HALF_SIZE = MAP_HALF_SIZE

        self.void_map: np.ndarray = None
        self.W: np.ndarray = None
        self.b: np.ndarray = None

    def real2grid(self, traj: np.ndarray) -> np.ndarray:
        if not type(traj) == np.ndarray:
            traj = np.array(traj)

        grid = ((traj - self.b) * self.W).astype(np.int32)
        return grid

    def pooling2D(self, maps: np.ndarray):
        """
        Apply MaxPooling on a batch of maps.

        :param maps: maps, shape = (batch, a, b)
        """
        maps = maps[..., np.newaxis]
        return POOLING_LAYER(maps).numpy()[..., 0]

    def map_exsits(self, pooling=False):
        if not pooling:
            if os.path.exists(self.path.format(self.FILE)):
                return True
            else:
                return False
        else:
            if os.path.exists(self.path.format(self.FILE_WITH_POOLING)):
                return True
            else:
                return False

    def build(self, agent: Agent,
              source: np.ndarray = None,
              *args, **kwargs):
        """
        Build a map for a specific agent.
        """
        raise NotImplementedError

    def build_all(self, agents: list[Agent],
                  source: np.ndarray = None,
                  *args, **kwargs):
        """
        Build maps for all agents and save them.
        """
        maps = []
        for agent in SecondaryBar(agents,
                                  manager=self.manager.manager,
                                  desc='Building Maps...'):
            maps.append(self.build(agent, source, *args, **kwargs))

        # save maps
        np.save(self.path.format(self.FILE), maps)

    def load(self, pooling=False) -> np.ndarray:
        """
        Load maps from the saved file.
        """
        maps = np.load(self.path.format(self.FILE), allow_pickle=True)

        if not pooling:
            return maps
        else:
            if not self.map_exsits(pooling=True):
                maps_pooling = self.pooling2D(np.array(maps))
                np.save(self.path.format(self.FILE_WITH_POOLING), maps_pooling)

            else:
                maps_pooling = np.load(self.path.format(self.FILE_WITH_POOLING),
                                       allow_pickle=True)

            return maps_pooling


class TrajMapManager(__BaseMapManager):

    def __init__(self, manager: BaseManager,
                 agents: list[Agent],
                 init_trajs: np.ndarray,
                 map_type: str,
                 base_path: str,
                 name: str = 'Trajectory Map Manager'):

        super().__init__(manager, map_type, base_path, name)

        self.FILE = 'trajMap.npy'
        self.FILE_WITH_POOLING = 'trajMap_pooling.npy'
        self.CONFIG_FILE = 'trajMap_configs.npy'

        self.GLOBAL_FILE = 'trajMap.png'
        self.GLOBAL_CONFIG_FILE = 'trajMap_configs.npy'

        # global trajectory map
        self.map: np.ndarray = None

        if self.map_exsits():
            pass
        else:
            self.init(init_trajs)
            self.build_all(agents)

    def init(self, init_trajs: np.ndarray):
        path_global_map = self.path.format(self.GLOBAL_FILE)
        if os.path.exists(path_global_map):
            self.map = self.load_global()
        else:
            self.map = self.build_global(init_trajs)

    def init_global(self, init_trajs: np.ndarray):
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

    def build_global(self, trajs: np.ndarray,
                     source: np.ndarray = None):

        if source is None:
            if self.void_map is None:
                self.void_map, self.W, self.b = self.init_global(trajs)

            source = self.void_map

        # build the global trajectory map
        source = source.copy()
        source = add(source,
                     self.real2grid(trajs),
                     amplitude=[1],
                     radius=7,
                     add_mask=MASK,
                     max_limit=False)

        source = np.minimum(source, 30)
        source = 1 - source / np.max(source)

        # save global trajectory map
        cv2.imwrite(self.path.format(self.GLOBAL_FILE), 255 * source)

        # save global map's configs
        np.save(self.path.format(self.GLOBAL_CONFIG_FILE),
                arr=dict(void_map=self.void_map,
                         W=self.W,
                         b=self.b),)

        return source

    def load_global(self):
        # load global trajectory map
        t_map = cv2.imread(self.path.format(self.GLOBAL_FILE))

        if t_map is None:
            raise FileNotFoundError

        t_map = (t_map[:, :, 0]).astype(np.float32)/255.0
        self.map = t_map
        
        # load global map's configs
        config_path = self.path.format(self.GLOBAL_CONFIG_FILE)
        if not os.path.exists(self.GLOBAL_CONFIG_FILE):
            self.log(f'Please delete the folder `{self.dir}` and' + 
                     ' re-run this program.', 'error')
            exit()

        config_dict = np.load(config_path, allow_pickle=True).tolist()

        self.void_map = config_dict['void_map']
        self.W = config_dict['W']
        self.b = config_dict['b']

    def build(self, agent: Agent,
              source: np.ndarray = None,
              *args, **kwargs):

        # Cut local trajectory map from the global map
        # Center point: the last observed point
        center_real = agent.traj[-1:, :]
        center_pixel = self.real2grid(center_real)
        local_map = cut(self.map, center_pixel, self.HALF_SIZE)[0]

        return local_map


class SocialMapManager(__BaseMapManager):

    def __init__(self, manager: BaseManager,
                 agents: list[Agent],
                 init_manager: TrajMapManager,
                 map_type: str,
                 base_path: str,
                 name: str = 'Social Map Manager'):

        super().__init__(manager, map_type, base_path, name)

        self.FILE = 'socialMap.npy'
        self.FILE_WITH_POOLING = 'socialMap_pooling.npy'
        self.CONFIG_FILE = 'socialMap_configs.npy'

        if self.map_exsits():
            pass
        else:
            self.init(init_manager)
            self.build_all(agents)

    def init(self, init_manager: TrajMapManager):
        self.void_map, self.W, self.b = [init_manager.void_map,
                                         init_manager.W,
                                         init_manager.b]

    def build(self, agent: Agent,
              source: np.ndarray = None,
              regulation=True,
              max_neighbor=15,
              *args, **kwargs) -> np.ndarray:
        """
        Build social map for a specific agent.
        TODO: Social maps for M-dimensional trajectories

        :param agent: target `Agent` object to calculate the map
        :param source: source map, default are zeros
        :param regulation: controls if scale the map into [0, 1]
        :param max_neighbor: the maximum number of neighbors to calculate
            the social map. Set it to a smaller value to speed up building
            on datasets thay contain more agents.
        """

        # build the global social map
        if type(source) == type(None):
            source = self.void_map

        source = source.copy()

        # Destination
        source = add(target_map=source,
                     grid_trajs=self.real2grid(agent.pred_linear),
                     amplitude=[-2],
                     radius=INTEREST_SIZE,
                     add_mask=MASK,
                     max_limit=False)

        # Interplay
        traj_neighbors = agent.pred_linear_neighbor
        amp_neighbors = []

        vec_target = agent.pred_linear[-1] - agent.pred_linear[0]
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

        source = add(target_map=source,
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

        # Get the local social map from the global map
        # center point: the last observed point
        center_real = agent.traj[-1:, :]
        center_pixel = self.real2grid(center_real)
        local_map = cut(source, center_pixel, self.HALF_SIZE)[0]

        return local_map


def calculate_cosine(vec1: np.ndarray,
                     vec2: np.ndarray):

    length1 = np.linalg.norm(vec1, axis=-1)
    length2 = np.linalg.norm(vec2, axis=-1)

    return (np.sum(vec1 * vec2, axis=-1) + 0.0001) / ((length1 * length2) + 0.0001)


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)


def activation(x: np.ndarray, a=1, b=1):
    return np.less_equal(x, 0) * a * x + np.greater(x, 0) * b * x


def add(target_map: np.ndarray,
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
    target_map = add_traj(target_map,
                          points, amps, radius,
                          MASKS[radius],
                          max_limit=max_limit)

    return target_map


def add_traj(source_map: np.ndarray,
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


def cut(target_map: np.ndarray,
        centers: np.ndarray,
        half_size: int) -> np.ndarray:
    """
    Cut several local maps from the target map.

    :param target_map: target map, shape = (a, b)
    :param centers: center positions (in grids), shape = (batch, 2)
    :param half_size: half size of the cut map
    """
    a, b = target_map.shape[-2:]
    centers = centers.astype(np.int32)

    # reshape to (batch, 2)
    if centers.ndim == 1:
        centers = centers[np.newaxis, :]

    centers = np.maximum(centers, half_size)
    centers = np.array([np.minimum(centers[:, 0], a - half_size),
                        np.minimum(centers[:, 1], b - half_size)]).T

    cuts = []
    for c in centers:
        cuts.append(target_map[c[0] - half_size: c[0] + half_size,
                               c[1] - half_size: c[1] + half_size])

    return np.array(cuts)
