"""
@Author: Conghao Wong
@Date: 2022-07-19 10:32:41
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-01 20:29:51
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np
from tqdm import tqdm

from ..__base import BaseObject
from ..args import BaseArgTable as Args
from ..utils import INIT_POSITION, MAP_HALF_SIZE, dir_check, load_from_plist
from .__agent import Agent
from .__maps import MapManager
from .__trajectory import Trajectory


class VideoClip():
    """
    VideoClip
    ---------
    Base structure for controlling each video dataset.

    Properties
    -----------------
    ```python
    >>> self.annpath        # dataset annotation file
    >>> self.anntype        # annotation type
    >>> self.dimension      # annotation dimension
    >>> self.matrix         # transfer matrix from real scales to pixels
    >>> self.name           # video clip name
    >>> self.paras          # [sample_step, frame_rate]
    >>> self.scale          # annotation scales
    >>> self.scale_vis      # scale when saving visualized images
    >>> self.video_path     # video path    
    ```
    """

    # Saving paths
    BASE_DIR = './dataset_original/dataset_configs'
    CONFIG_FILE = os.path.join(BASE_DIR, '{}', 'subsets', '{}.plist')

    def __init__(self, name: str,
                 dataset: str,
                 annpath: str = None,
                 order: tuple[int, int] = None,
                 paras: tuple[int, int] = None,
                 video_path: str = None,
                 matrix: list[float] = None,
                 datasetInfo=None,
                 *args, **kwargs):

        self.__name = name
        self.__annpath = annpath
        self.__order = order
        self.__paras = paras
        self.__video_path = video_path
        self.__matrix = matrix
        self.__dataset = dataset

        self.CONFIG_FILE = self.CONFIG_FILE.format(self.dataset, self.name)

        # make dirs
        dirs = [self.CONFIG_FILE]
        for d in dirs:
            _dir = os.path.dirname(d)
            dir_check(_dir)

    def get(self):
        plist_path = self.CONFIG_FILE

        try:
            dic = load_from_plist(plist_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}` NOT FOUND.'.format(plist_path))

        return VideoClip(**dic)

    @property
    def dataset(self) -> str:
        """
        Name of the dataset.
        """
        return self.__dataset

    @property
    def name(self):
        """
        Name of the video clip.
        """
        return self.__name

    @property
    def annpath(self) -> str:
        """
        Path of the annotation file. 
        """
        return self.__annpath

    @property
    def order(self) -> list[int]:
        """
        X-Y order in the annotation file.
        """
        return self.__order

    @property
    def paras(self) -> tuple[int, int]:
        """
        [sample_step, frame_rate]
        """
        return self.__paras

    @property
    def video_path(self) -> str:
        """
        Path of the video file.
        """
        return self.__video_path

    @property
    def matrix(self) -> list[float]:
        """
        transfer weights from real scales to pixels.
        """
        return self.__matrix


class TrajMapNotFoundError(FileNotFoundError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class VideoClipManager(BaseObject):
    """
    VideoClipManager
    ----------------
    Manage all training data from one video clip.

    Properties
    ----------
    ```python
    >>> self.args   # args
    >>> self.name   # name
    >>> self.dataset # name of the video dataset
    ```

    Public Methods
    --------------
    ```python
    # Sample train data from dataset
    (method) sample_train_data: (self: DatasetManager) -> list[agent_type]

    # Load dataset files
    (method) load_data: (self: DatasetManager) -> Any
    ```
    """

    def __init__(self, args: Args, name: str,
                 temp_dir: str,
                 custom_list: list[np.ndarray] = []):

        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.name = name
        self.path = temp_dir

        self.info = VideoClip(name=name, dataset=self.dataset).get()
        self.custom_list = custom_list
        self.agent_count = None
        self.trajectories: list[Trajectory] = None
        self.bar: tqdm = None

    def load_dataset(self, file_name: str):
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

        :param file_name: name of the annatation file
        """

        data = np.genfromtxt(file_name, dtype=np.str, delimiter=',')

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
            agent_dict[name] = _dat[:, :-2].astype(np.float64)

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
            agent_dict, frame_ids = self.load_dataset(self.info.annpath)
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

    def sample_train_data(self) -> list[Agent]:
        """
        Sampling train samples from trajectories.
        """

        if self.trajectories is None:
            self.make_trajectories()

        sample_rate, frame_rate = self.info.paras
        frame_step = int(0.4 / (sample_rate / frame_rate))
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

        return train_samples

    def make_maps(self, agents: list[Agent],
                  map_type: str,
                  base_path: str,
                  save_map_file: str = None,
                  save_social_file: str = 'socialMap.npy',
                  save_para_file: str = 'para.txt',
                  save_centers_file: str = 'centers.txt'):
        """
        Make maps for input agents, and save them in the numpy format.

        :param agents: a list of agents that ready to calculate maps
        :param base_path: base folder to save the map and map parameters
        :param load_map_file: file name for the saved trajectory map (`.jpg` or `.png`).
        default is `None`. When this item is `None`, MapManager will build
        trajectory maps according to trajectories of the input agents.
        :param save_map_file: file name to save the built traj map
        :param save_social_file: file name to save the social map (already cut)
        :param save_para_file: file name to save the map parameters
        :param save_centers_file: path to save the centers
        """

        map_manager = MapManager(self.args, map_type, agents)

        if save_map_file:
            traj_map = map_manager.build_guidance_map(
                agents=agents,
                save=os.path.join(base_path, save_map_file))

        social_maps = []
        centers = []
        agent_count = len(agents)
        for index, agent in enumerate(agents):
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
