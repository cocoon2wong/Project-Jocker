"""
@Author: Conghao Wong
@Date: 2022-08-03 10:50:46
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-15 10:36:57
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..args import Args
from ..base import BaseObject, SecondaryBar
from ..utils import MAP_HALF_SIZE
from .__agent import Agent
from .__maps import MapManager
from .__picker import Picker


class TrajMapNotFoundError(FileNotFoundError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class AgentManager(BaseObject):
    """
    AgentManager
    ---
    Structure to manage several `Agent` objects.

    Public Methods
    ---
    ```python
    # concat agents to this `AgentManager`
    (method) append: (self: Self@AgentManager, target: Any) -> None

    # set inputs and outputs
    (method) set: (self: Self@AgentManager, dimension: int, 
                   inputs_type: list[str],
                   labels_type: list[str]) -> None

    # get inputs
    (method) get_inputs: (self: Self@AgentManager) -> list[Tensor]

    # get labels
    (method) get_labels: (self: Self@AgentManager) -> list[Tensor]

    # make inputs and labels into dataset
    (method) make_dataset: (self: Self@AgentManager, 
                            shuffle: bool = False) -> DatasetV2

    # save all agents' data
    (method) save: (self: Self@AgentManager, save_dir: str) -> None

    # load from saved agents' data
    (method) load: (cls: Type[Self@AgentManager], path: str) -> AgentManager
    ```

    Context Map Methods
    ---
    ```python
    # make context maps (both social maps and guidance maps) and save
    (method) make_maps: (self: Self@AgentManager,
                         map_type: str, base_path: str, 
                         save_map_file: str = None, 
                         save_social_file: str = 'socialMap.npy', 
                         save_para_file: str = 'para.txt', 
                         save_centers_file: str = 'centers.txt') -> None

    #  load context maps from the saved files
    (method) load_maps: (self: Self@AgentManager,
                         base_path: str, map_file: str,
                         social_file: str,
                         para_file: str,
                         centers_file: str) -> None
    ```
    """

    def __init__(self, args: Args, agents: list[Agent]):

        super().__init__()
        self.args = args
        self.agents = agents

        self.model_inputs = None
        self.model_labels = None
        self.picker: Picker = None

    def append(self, target):
        self.agents += target.agents

    def set_picker(self, datasetType: str, predictionType: str):
        self.picker = Picker(datasetType, predictionType)
        for agent in self.agents:
            agent.picker = self.picker

    def set_types(self, inputs_type: list[str], labels_type: list[str]):
        """
        Set type of model inputs and outputs.

        :param inputs_type: a list of `str`, accept `'TRAJ'`, `'MAPPARA'`,
            `'MAP'`, `'DEST'`, and `'GT'`
        :param labels_type: a list of `str`, accept `'GT'` and `'DEST'`
        """

        self.model_inputs = inputs_type
        self.model_labels = labels_type

    def get_inputs(self) -> list[tf.Tensor]:
        """
        Get all model inputs from agents.
        """
        return [self._get(T) for T in self.model_inputs]

    def get_labels(self) -> list[tf.Tensor]:
        """
        Get all model labels from agents.
        """
        return [self._get(T) for T in self.model_labels]

    def get_inputs_and_labels(self) -> list[tf.Tensor]:
        """
        Get model inputs and labels (only trajectories) from all agents.
        """
        inputs = self.get_inputs()
        labels = self.get_labels()[0]

        inputs.append(labels)
        return tuple(inputs)

    def make_dataset(self, shuffle=False) -> tf.data.Dataset:
        """
        Get inputs from all agents and make the `tf.data.Dataset`
        object. Note that the dataset contains both model inputs
        and labels.
        """
        data = self.get_inputs_and_labels()
        dataset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            dataset = dataset.shuffle(
                len(dataset),
                reshuffle_each_iteration=True
            )

        return dataset

    def save(self, save_dir: str):
        """
        Save data of all agents.

        :param save_dir: directory to save agent data
        """
        save_dict = {}
        for index, agent in enumerate(self.agents):
            save_dict[str(index)] = agent.zip_data()

        np.savez(save_dir, **save_dict)

    @classmethod
    def load(cls, args: Args, path: str):
        """
        Load agents' data from saved file.

        :param path: file path of the saved data
        """
        save_dict: dict = np.load(path, allow_pickle=True)

        if (v := save_dict['0'].tolist()['__version__']) < (v1 := Agent.__version__):
            cls.log((f'Saved agent managers\' version is {v}, ' +
                     f'which is lower than current {v1}. Please delete ' +
                     'them and re-run this program, or there could ' +
                     'happen something wrong.'),
                    level='error')

        return AgentManager(args, [Agent().load_data(v.tolist())
                                   for v in save_dict.values()])

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

        map_manager = MapManager(self.args, map_type, self._get_obs_trajs())

        if save_map_file:
            traj_map = map_manager.build_guidance_map(
                trajs=self._get_obs_trajs(),
                save=os.path.join(base_path, save_map_file))

        social_maps = []
        centers = []
        for agent in SecondaryBar(self.agents,
                                  bar=self.bar,
                                  desc='Building Maps...'):

            centers.append(agent.traj[-1:, :])
            social_maps.append(map_manager.build_social_map(agent))

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
                  centers_file: str):
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

    def _get_obs_trajs(self) -> np.ndarray:
        return np.array([a.traj for a in self.agents])

    def _get(self, type_name: str) -> tf.Tensor:
        """
        Get model inputs from a list of `Agent`-like objects.

        :param type_name: inputs names, accept `'TRAJ'`, 
            `'MAP'`, `'MAPPARA'`, `'DEST'`, and `'GT'`
        :return inputs: a tensor of stacked inputs
        """
        if type_name == 'TRAJ':
            call = _get_obs_traj
        elif type_name == 'MAP':
            call = _get_context_map
        elif type_name == 'MAPPARA':
            call = _get_context_map_paras
        elif type_name == 'DEST':
            call = _get_dest_traj
        elif type_name == 'GT':
            call = _get_gt_traj
        else:
            raise ValueError(type_name)

        return call(self.agents)


def _get_obs_traj(input_agents: list[Agent]) -> tf.Tensor:
    """
    Get observed trajectories from agents.

    :param input_agents: a list of input agents, type = `list[Agent]`
    :return inputs: a Tensor of observed trajectories
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare trajectories...'):
        inputs.append(agent.traj)
    return tf.cast(inputs, tf.float32)


def _get_gt_traj(input_agents: list[Agent],
                 destination=False) -> tf.Tensor:
    """
    Get groundtruth trajectories from agents.

    :param input_agents: a list of input agents, type = `list[Agent]`
    :return inputs: a Tensor of gt trajectories
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare groundtruth...'):
        if destination:
            inputs.append(np.expand_dims(agent.groundtruth[-1], 0))
        else:
            inputs.append(agent.groundtruth)

    return tf.cast(inputs, tf.float32)


def _get_dest_traj(input_agents: list[Agent]) -> tf.Tensor:
    return _get_gt_traj(input_agents, destination=True)


def _get_context_map(input_agents: list[Agent]) -> tf.Tensor:
    """
    Get context map from agents.

    :param input_agents: a list of input agents, type = `list[Agent]`
    :return inputs: a Tensor of maps
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare maps...'):
        inputs.append(agent.Map)
    return tf.cast(inputs, tf.float32)


def _get_context_map_paras(input_agents: list[Agent]) -> tf.Tensor:
    """
    Get parameters of context map from agents.

    :param input_agents: a list of input agents, type = `list[Agent]`
    :return inputs: a Tensor of map paras
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare maps...'):
        inputs.append(agent.real2grid)
    return tf.cast(inputs, tf.float32)
