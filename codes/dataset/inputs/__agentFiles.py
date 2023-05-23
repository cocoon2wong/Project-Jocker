"""
@Author: Conghao Wong
@Date: 2023-05-19 16:05:54
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-23 11:18:18
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np

from ...base import BaseManager
from ..__splitManager import Clip
from ..trajectories import Agent
from .__baseInputManager import BaseInputManager
from .__trajectoryManager import TrajectoryManager


class AgentFilesManager(BaseInputManager):
    """
    AgentFilesManager
    ---
    A manager to save processed agent files.

    - Load items: A list of agents (type is `list[Agent]`) to save;
    - Run items: Load agents and save them into `npz` files.
        If the saved file exists, it will load these files into agents.
    """

    def __init__(self, manager: BaseManager,
                 name='Agent Files Manager'):

        super().__init__(manager, name)

    def get_temp_file_path(self, clip: Clip) -> str:
        base_dir = clip.temp_dir
        if (self.args.obs_frames, self.args.pred_frames) == (8, 12):
            f_name = 'agent'
        else:
            f_name = f'agent_{self.args.obs_frames}to{self.args.pred_frames}'

        endstring = '' if self.args.step == 4 else str(self.args.step)
        f_name = f_name + endstring + '.npz'
        return os.path.join(base_dir, f_name)

    def run(self, clip: Clip, agents: list[Agent] = None,
            *args, **kwargs) -> list[Agent]:

        return super().run(clip=clip, agents=agents, *args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        agents = self.manager.get_member(
            TrajectoryManager).run(self.working_clip)

        save_dict = {}
        for index, agent in enumerate(agents):
            save_dict[str(index)] = agent.zip_data()

        np.savez(self.temp_file, **save_dict)

    def load(self, *args, **kwargs) -> list:
        saved: dict = np.load(self.temp_file, allow_pickle=True)

        if not len(saved):
            self.log(f'Please delete file `{self.temp_file}` and re-run the program.',
                     level='error', raiseError=FileNotFoundError)

        if (v := saved['0'].tolist()['__version__']) < (v1 := Agent.__version__):
            self.log((f'Saved agent managers\' version is {v}, ' +
                      f'which is lower than current {v1}. Please delete' +
                      ' them and re-run this program, or there could' +
                      ' happen something wrong.'),
                     level='error')

        return [Agent().load_data(v.tolist()) for v in saved.values()]
