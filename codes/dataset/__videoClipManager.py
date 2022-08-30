"""
@Author: Conghao Wong
@Date: 2022-08-03 09:30:41
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-30 11:29:28
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np

from ..args import Args
from ..base import BaseObject
from ..utils import INIT_POSITION, TEMP_PATH, dir_check
from .__agentManager import AgentManager
from .__trajectory import Trajectory
from .__videoClip import VideoClip


class VideoClipManager(BaseObject):
    """
    VideoClipManager
    ----------------
    Manage all training data from one video clip.


    Public Methods
    --------------
    ```python
    # Load trajectory data from the annotation txt file
    (method) load_dataset: (self: Self@VideoClipManager) -> tuple[dict, list]

    # Process metadata of a video clip (like csv dataset files)
    (method) process_metadata: (self: Self@VideoClipManager) 
        -> tuple[ndarray, ndArray, list, list]

    # Make trajectories from the processed dataset files
    (method) make_trajectories: (self: Self@VideoClipManager)
        -> Self@VideoClipManager

    # Sampling train agents from trajectories
    (method) sample_train_data: (self: Self@VideoClipManager) -> AgentManager
    ```
    """

    def __init__(self, args: Args, name: str,
                 custom_list: list[np.ndarray] = []):

        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.name = name

        self.path = TEMP_PATH.format(self.dataset)
        self.info = VideoClip(name=name, dataset=self.dataset).get()

        self.custom_list = custom_list
        self.agent_count = None
        self.trajectories: list[Trajectory] = None

    def load_dataset(self):
        """
        Load trajectory data from the annotation txt file.
        Data format of the `ann.txt`:
        It is a matrix with the shape = `(N, M)`, where
        - `N` is the number of records in the file;
        - `M` is the length of each record.

        A record may contains several items, where
        - `item[0]`: frame name (or called the frame id);
        - `item[1]`: agent name (or called the agent id);
        - `item[2:M-1]`: dataset records, like coordinates, 
            bounding boxes, and other type of trajectory series.
        - `item[M-1]`: type of the agent
        """

        data = np.genfromtxt(self.info.annpath, str, delimiter=',')
        anndim = self.info.datasetInfo.dimension

        agent_dict = {}
        agent_names = np.unique(agent_order := data.T[1])

        try:
            agent_ids = [int(n.split('_')[0]) for n in agent_names]
            agent_order = np.argsort(agent_ids)
        except:
            agent_order = np.arange(len(agent_names))

        for agent_index in agent_order:
            name = agent_names[agent_index]
            index = np.where(data.T[1] == name)[0]
            _dat = np.delete(data[index], 1, axis=1)
            agent_dict[name] = _dat[:, :anndim+1].astype(np.float64)

        frame_ids = list(set(data.T[0].astype(np.int32)))
        frame_ids.sort()

        return agent_dict, frame_ids

    def process_metadata(self):
        """
        Process metadata of a video clip (like csv dataset 
        files) into numpy ndarray.
        """
        # make directories
        b = dir_check(os.path.join(dir_check(self.path), self.name))
        npy_path = os.path.join(b, 'data.npz')

        # load from saved files
        if os.path.exists(npy_path):
            dat = np.load(npy_path, allow_pickle=True)
            matrix = dat['matrix']
            neighbor_indexes = dat['neighbor_indexes']
            frame_ids = dat['frame_ids']
            agent_names = dat['person_ids']

        # or start processing and then saving
        else:
            agent_dict, frame_ids = self.load_dataset()
            agent_names = list(agent_dict.keys())

            p = len(agent_names)
            f = len(frame_ids)

            # agent_name -> agent_index
            name_dict = dict(zip(agent_names, np.arange(p)))

            # frame_id -> frame_index
            frame_dict = dict(zip(frame_ids, np.arange(f)))

            # init the matrix
            dim = agent_dict[agent_names[0]].shape[-1] - 1
            matrix = INIT_POSITION * np.ones([f, p, dim])

            for name, index in name_dict.items():

                p = '{}%'.format((index+1)*100/len(name_dict))
                self.update_timebar(self.bar, 'Processing dataset: ' + p)

                frame_id = agent_dict[name].T[0].astype(np.int32)
                frame_index = [frame_dict[fi] for fi in frame_id]

                matrix[frame_index, index, :] = agent_dict[name][:, 1:]

            neighbor_indexes = np.array([
                np.where(np.not_equal(data, INIT_POSITION))[0]
                for data in matrix[:, :, 0]], dtype=object)

            np.savez(npy_path,
                     neighbor_indexes=neighbor_indexes,
                     matrix=matrix,
                     frame_ids=frame_ids,
                     person_ids=agent_names)

        self.agent_count = matrix.shape[1]
        self.frame_number = matrix.shape[0]

        return neighbor_indexes, matrix, frame_ids, agent_names

    def make_trajectories(self):
        """
        Make trajectories from the processed dataset files.
        """
        if len(self.custom_list) == 4:
            self.nei_indexes, self.matrix, self.frame_ids, self.agent_names = self.custom_list
        else:
            self.nei_indexes, self.matrix, self.frame_ids, self.agent_names = self.process_metadata()

        trajs = []
        for agent_index in range(self.agent_count):
            trajs.append(Trajectory(agent_id=self.agent_names[agent_index],
                                    trajectory=self.matrix[:, agent_index, :],
                                    neighbors=self.nei_indexes,
                                    frames=self.frame_ids,
                                    init_position=INIT_POSITION,
                                    dimension=self.args.dim))

        self.trajectories = trajs
        return self

    def sample_train_data(self) -> AgentManager:
        """
        Sampling train agents from trajectories.
        """

        if self.trajectories is None:
            self.make_trajectories()

        sample_rate, frame_rate = self.info.paras
        frame_step = int(self.args.interval / (sample_rate / frame_rate))
        train_samples = []

        for agent_index in range(self.agent_count):

            # update timebar
            p = '{}%'.format((agent_index + 1)*100//self.agent_count)
            self.update_timebar(self.bar, 'Prepare train data: ' + p)

            trajectory = self.trajectories[agent_index]
            start_frame = trajectory.start_frame
            end_frame = trajectory.end_frame

            for p in range(start_frame, end_frame, self.args.step * frame_step):
                # Normal mode
                if self.args.pred_frames > 0:
                    if p + (self.args.obs_frames + self.args.pred_frames) * frame_step > end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = p + (self.args.obs_frames +
                               self.args.pred_frames) * frame_step

                # Infinity mode, only works for destination models
                elif self.args.pred_frames == -1:
                    if p + (self.args.obs_frames + 1) * frame_step > end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = end_frame

                else:
                    raise ValueError(
                        '`pred_frames` should be a positive integer or -1.')

                train_samples.append(trajectory.sample(start_frame=p,
                                                       obs_frame=obs,
                                                       end_frame=end,
                                                       matrix=self.matrix,
                                                       frame_step=frame_step,
                                                       add_noise=False))

        return AgentManager(self.args, train_samples)
