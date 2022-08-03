"""
@Author: Conghao Wong
@Date: 2022-08-03 09:34:55
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-03 12:01:25
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import random
from typing import Union

from ..__base import BaseObject
from ..args import BaseArgTable as Args
from ..utils import dir_check
from .__agentManager import AgentManager, TrajMapNotFoundError
from .__videoClipManager import VideoClipManager
from .__videoDataset import Dataset


class DatasetManager(BaseObject):
    """
    DatasetsManager
    ---------------
    Manage all prediction training data.
    """

    def __init__(self, args: Args):
        super().__init__()

        self.args = args
        self.info = Dataset(args.dataset, args.split)

        self.model_inputs = None
        self.model_labels = None

    def set(self, inputs_type: list[str],
            labels_type: list[str] = None):

        self.model_inputs = inputs_type
        if labels_type is not None:
            self.model_labels = labels_type

    def _load_from_videoClips(self, video_clips: list[VideoClipManager],
                              mode='test') -> AgentManager:
        """
        Make or load train files to get train agents.
        (a list of agent managers, type = `Agent`)

        :param video_clips: a list of video clip managers (`VideoClipManager`)
        :return all_agents: a list of train agents (`AgentManager`)
        """
        all_agents = AgentManager([])

        if mode == 'train':
            random.shuffle(video_clips)

        self.bar = self.timebar(video_clips)
        for clip in (self.bar):
            # assign time bar
            s = 'Prepare {} data in `{}`...'.format(mode, clip.name)
            self.update_timebar(self.bar, s, pos='start')
            clip.bar = self.bar

            base_dir = os.path.join(clip.path, clip.name)
            if (self.args.obs_frames, self.args.pred_frames) == (8, 12):
                f_name = 'agent'
            else:
                f_name = 'agent_{}to{}'.format(self.args.obs_frames,
                                               self.args.pred_frames)

            endstring = '' if self.args.step == 4 else str(self.args.step)
            f_name = f_name + endstring + '.npz'
            data_path = os.path.join(base_dir, f_name)

            if not os.path.exists(data_path):
                agents = clip.sample_train_data()
                agents.save(data_path)
            else:
                agents = AgentManager.load(data_path)

            if self.args.use_maps:
                map_path = dir_check(data_path.split('.np')[0] + '_maps')
                map_file = ('trajMap.png' if not self.args.use_extra_maps
                            else 'trajMap_load.png')
                map_type = self.info.type

                try:
                    agents.load_maps(map_path,
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
                    agents.make_maps(map_type, map_path,
                                     save_map_file='trajMap.png',
                                     save_social_file='socialMap.npy',
                                     save_para_file='para.txt',
                                     save_centers_file='centers.txt')

                    agents.load_maps(map_path,
                                     map_file=map_file,
                                     social_file='socialMap.npy',
                                     para_file='para.txt',
                                     centers_file='centers.txt')

            all_agents.append(agents)

        all_agents.set(dimension=self.args.dim,
                       inputs_type=self.model_inputs,
                       labels_type=self.model_labels)
        return all_agents

    def load(self, clips: Union[str, list[str]], mode: str) -> Union[AgentManager, tuple[AgentManager, AgentManager]]:
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

            return [train_agents, test_agents]

        else:
            if type(clips) == str:
                clips = [clips]

            dms = [VideoClipManager(self.args, d) for d in clips]
            return self._load_from_videoClips(dms, mode=mode)
