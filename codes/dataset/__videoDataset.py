"""
@Author: Conghao Wong
@Date: 2022-07-19 11:19:58
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-19 14:06:21
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Union

import cv2
import numpy as np

from ..__base import BaseObject
from ..args import BaseArgTable as Args
from ..utils import MAP_HALF_SIZE, dir_check, load_from_plist, TEMP_PATH
from .__agent import Agent
from .__maps import MapManager
from .__videoClip import TrajMapNotFoundError, VideoClipManager


class Dataset():
    """
    Dataset
    -------
    Manage a full trajectory prediction dataset.
    A dataset may contains several video clips.

    """

    def __init__(self, dataset: str, split: str,
                 root_dir='./dataset_configs'):
        """
        init

        :param dataset: name of the image dataset
        :param split: split name of the dataset
        :param root_dir: dataset config folder
        """
        split_path = os.path.join(root_dir,
                                  dataset,
                                  '{}.plist'.format(split))
        try:
            dic = load_from_plist(split_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}` NOT FOUND.'.format(split_path))

        self.train_sets: list[str] = dic['train']
        self.test_sets: list[str] = dic['test']
        self.val_sets: list[str] = dic['val']

        self._anntype: str = dic['anntype']
        self._dataset: str = dic['dataset']
        self._dimension: int = dic['dimension']
        self._scale: float = dic['scale']
        self._type: str = dic['type']

    @property
    def anntype(self):
        """
        Type of annotations in this video clip.
        canbe `'coordinate'`, `'boundingbox'`, ...
        """
        return self._anntype

    @property
    def dataset(self) -> str:
        """
        Name of the video dataset.
        For example, `ETH-UCY` or `SDD`.
        """
        return self._dataset

    @property
    def dimension(self) -> int:
        """
        Maximum dimension of trajectories recorded in this dataset.
        For example, `(x, y)` -> `dimension = 2`.
        """
        return self._dimension

    @property
    def scale(self) -> float:
        """
        Maximum pixel length of the images.
        For example, `weights = 1920` when `(H, W) = (1920, 1080)`.
        """
        return self._scale

    @property
    def type(self) -> str:
        """
        Type of the dataset, canbe `'pixel'` or `'meter'`.
        """
        return self._type


class DatasetManager(BaseObject):
    """
    DatasetsManager
    ---------------
    Manage all prediction training data.

    Public Methods
    --------------
    ```python
    # Prepare train agents from `DatasetManager`s
    (method) load_fromManagers: (self: DatasetsManager, dataset_managers: list[DatasetManager], mode='test') -> list[Agent]

    # Save and load agents' data
    (method) zip_and_save: (save_dir, agents: list[Agent]) -> None
    (method) load_and_unzip: (cls: Type[DatasetsManager], save_dir) -> list[Agent]
    ```
    """

    def __init__(self, args: Args):
        super().__init__()

        self.args = args
        self.info = Dataset(args.dataset, args.split)

    def load_from_videoClips(self, video_clips: list[VideoClipManager],
                             mode='test') -> list[Agent]:
        """
        Make or load train files to get train agents.
        (a list of agent managers, type = `Agent`)

        :param video_clips: a list of video clip managers (`VideoClipManager`)
        :return all_agents: a list of train agents (`Agent`)
        """
        all_agents = []
        count = 1
        total = len(video_clips)

        for clip in video_clips:
            print('({}/{})  Prepare test data in `{}`...'.format(
                count, total, clip.name))

            base_dir = os.path.join(clip.path, clip.name)
            if (self.args.obs_frames, self.args.pred_frames) == (8, 12):
                f_name = 'agent'
            else:
                f_name = 'agent_{}to{}'.format(self.args.obs_frames,
                                               self.args.pred_frames)

            endstring = '' if self.args.step == 4 else self.args.step
            f_name += '{}.npz'.format(endstring)
            data_path = os.path.join(base_dir, f_name)

            if not os.path.exists(data_path):
                agents = clip.sample_train_data()
                self.zip_and_save(data_path, agents)
            else:
                agents = self.load_and_unzip(data_path)

            self.log('Successfully load train agents from `{}`'.format(data_path))

            if self.args.use_maps:
                map_path = dir_check(data_path.split('.np')[0] + '_maps')
                map_file = ('trajMap.png' if not self.args.use_extra_maps
                            else 'trajMap_load.png')
                map_type = self.info.type

                try:
                    agents = self.load_maps(map_path, agents,
                                            map_file=map_file,
                                            social_file='socialMap.npy',
                                            para_file='para.txt',
                                            centers_file='centers.txt')

                except TrajMapNotFoundError:
                    path = os.path.join(map_path, map_file)
                    self.log(s := ('Trajectory map `{}`'.format(path) +
                                   ' not found, stop running...'),
                             level='error')
                    exit()

                except:
                    self.log('Load maps failed, start re-making...')

                    clip.make_maps(agents, map_type, map_path,
                                   save_map_file='trajMap.png',
                                   save_social_file='socialMap.npy',
                                   save_para_file='para.txt',
                                   save_centers_file='centers.txt')

                    agents = self.load_maps(map_path, agents,
                                            map_file=map_file,
                                            social_file='socialMap.npy',
                                            para_file='para.txt',
                                            centers_file='centers.txt')

                self.log('Successfully load maps from `{}`.'.format(map_path))

            all_agents += agents
            count += 1

        return all_agents

    def zip_and_save(self, save_dir, agents: list[Agent]):
        save_dict = {}
        for index, agent in enumerate(agents):
            save_dict[str(index)] = agent.zip_data()
        np.savez(save_dir, **save_dict)

    def load_and_unzip(self, save_dir) -> list[Agent]:
        save_dict = np.load(save_dir, allow_pickle=True)

        if save_dict['0'].tolist()['__version__'] < Agent.__version__:
            self.log(('Saved agent managers\' version is {}, ' +
                      'which is lower than current {}. Please delete ' +
                      'them and re-run this program, or there could ' +
                      'happen something wrong.').format(save_dict['0'].tolist()['__version__'],
                                                        Agent.__version__),
                     level='error')

        return [Agent(self.args.dim).load_data(save_dict[key].tolist()) for key in save_dict.keys()]

    def load(self, clips: Union[str, list[str]],
             mode: str):
        """
        Load train samples in sub-datasets (i.e., video clips).

        :param clips: clips to load. Set it to `'auto'` to load train agents
        :param mode: load mode, canbe `'test'` or `'train'`
        :param datasetInfo: dataset infomation. It should be given when
            `dataset` is not `'auto'`.

        :return agents: loaded agents. It returns a list of `[train_agents, test_agents]` when `mode` is `'train'`.
        """

        if clips == 'auto':
            train_agents = self.load(self.info.train_sets, mode='train')
            test_agents = self.load(self.info.test_sets, mode='test')

            return train_agents, test_agents

        else:
            if type(clips) == str:
                clips = [clips]

            dms = [VideoClipManager(self.args, d, temp_dir=TEMP_PATH) for d in clips]
            return self.load_from_videoClips(dms, mode=mode)

    def load_maps(self, base_path: str,
                  agents: list[Agent],
                  map_file: str,
                  social_file: str,
                  para_file: str,
                  centers_file: str) -> list[Agent]:
        """
        Load maps from the base folder

        :param base_path: base save folder
        :param agents: agents to assign maps
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

        for agent, t_map, s_map in zip(agents, traj_map_cut, social_map):
            Agent.set_map(agent, 0.5*t_map + 0.5*s_map, para)

        return agents
