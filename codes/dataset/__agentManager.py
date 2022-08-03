"""
@Author: Conghao Wong
@Date: 2022-08-03 10:50:46
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-03 14:38:48
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import numpy as np
import tensorflow as tf

from ..__base import BaseObject
from ..utils import MAP_HALF_SIZE
from .__agent import Agent
from .__io import get_inputs_by_type as get
from .__maps import MapManager


class TrajMapNotFoundError(FileNotFoundError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class AgentManager(BaseObject):

    def __init__(self, agents: list[Agent]):

        super().__init__()
        self.agents = agents

        self.model_inputs = None
        self.model_labels = None
        self.dimension = None

    def append(self, target):
        self.agents += target.agents

    def set(self, dimension: int,
            inputs_type: list[str],
            labels_type: list[str]):
        
        self.model_inputs = inputs_type
        self.model_labels = labels_type

        for agent in self.agents:
            agent.dim = dimension

    def get_inputs(self) -> list[tf.Tensor]:
        return [get(self.agents, T) for T in self.model_inputs]

    def get_labels(self) -> list[tf.Tensor]:
        return [get(self.agents, T) for T in self.model_labels]

    def get_inputs_and_labels(self) -> list[tf.Tensor]:
        inputs = self.get_inputs()
        labels = self.get_labels()[0]

        inputs.append(labels)
        return tuple(inputs)

    def make_dataset(self, shuffle=False) -> tf.data.Dataset:
        data = self.get_inputs_and_labels()
        dataset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            dataset = dataset.shuffle(
                len(dataset), 
                reshuffle_each_iteration=True
            )

        return dataset


    def save(self, save_dir: str):
        save_dict = {}
        for index, agent in enumerate(self.agents):
            save_dict[str(index)] = agent.zip_data()

        np.savez(save_dir, **save_dict)

    @classmethod
    def load(cls, path: str):
        save_dict = np.load(path, allow_pickle=True)

        if save_dict['0'].tolist()['__version__'] < Agent.__version__:
            cls.log(('Saved agent managers\' version is {}, ' +
                      'which is lower than current {}. Please delete ' +
                      'them and re-run this program, or there could ' +
                      'happen something wrong.').format(save_dict['0'].tolist()['__version__'],
                                                        Agent.__version__),
                     level='error')

        return AgentManager([Agent().load_data(save_dict[key].tolist()) for key in save_dict.keys()])

    def make_maps(self, map_type: str,
                  base_path: str,
                  save_map_file: str = None,
                  save_social_file: str = 'socialMap.npy',
                  save_para_file: str = 'para.txt',
                  save_centers_file: str = 'centers.txt'):
        """
        Make maps for input agents, and save them in the numpy format.

        :param base_path: base folder to save the map and map parameters
        :param load_map_file: file name for the saved trajectory map (`.jpg` or `.png`).
        default is `None`. When this item is `None`, MapManager will build
        trajectory maps according to trajectories of the input agents.
        :param save_map_file: file name to save the built traj map
        :param save_social_file: file name to save the social map (already cut)
        :param save_para_file: file name to save the map parameters
        :param save_centers_file: path to save the centers
        """

        map_manager = MapManager(self.args, map_type, self.agents)

        if save_map_file:
            traj_map = map_manager.build_guidance_map(
                agents=self.agents,
                save=os.path.join(base_path, save_map_file))

        social_maps = []
        centers = []
        agent_count = len(self.agents)
        for index, agent in enumerate(self.agents):
            centers.append(agent.traj[-1:, :])
            social_maps.append(map_manager.build_social_map(
                target_agent=agent,
                traj_neighbors=agent.get_pred_traj_neighbor_linear()))

            # update timebar
            p = '{}%'.format((index+1)*100//agent_count)
            self.update_timebar(self.bar, 'Building Maps: ' + p)

        social_maps = np.array(social_maps)  # (batch, a, b)

        centers = np.concatenate(centers, axis=0)
        centers = map_manager.real2grid(centers)
        cuts = map_manager.cut_map(social_maps,
                                   centers,
                                   MAP_HALF_SIZE)
        paras = map_manager.real2grid_paras

        np.savetxt(os.path.join(base_path, save_centers_file), centers)
        np.savetxt(os.path.join(base_path, save_para_file), paras)
        np.save(os.path.join(base_path, save_social_file), cuts)

    def load_maps(self, base_path: str,
                  map_file: str,
                  social_file: str,
                  para_file: str,
                  centers_file: str) -> list[Agent]:
        """
        Load maps from the base folder

        :param base_path: base save folder
        :param map_file: file name for traj maps, support `.jpg` or `.png`
        :param social_file: file name for social maps, support `.npy`
        :param para_file: file name for map parameters, support `.txt`
        :param centers_file: file name for centers, support `.txt`

        :return agents: agents with maps
        """
        traj_map = cv2.imread(os.path.join(base_path, map_file))

        if traj_map is None:
            if self.args.use_extra_maps:
                raise TrajMapNotFoundError
            else:
                raise FileNotFoundError

        traj_map = (traj_map[:, :, 0]).astype(np.float32)/255.0

        social_map = np.load(os.path.join(
            base_path, social_file), allow_pickle=True)
        para = np.loadtxt(os.path.join(base_path, para_file))
        centers = np.loadtxt(os.path.join(base_path, centers_file))

        batch_size = len(social_map)
        traj_map = np.repeat(traj_map[np.newaxis, :, :], batch_size, axis=0)
        traj_map_cut = MapManager.cut_map(traj_map,
                                          centers,
                                          MAP_HALF_SIZE)

        for agent, t_map, s_map in zip(self.agents, traj_map_cut, social_map):
            agent.set_map(0.5*t_map + 0.5*s_map, para)

  