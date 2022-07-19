"""
@Author: Conghao Wong
@Date: 2022-07-19 10:32:41
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-19 14:28:00
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np

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

    def __init__(self, name: str,
                 annpath: str,
                 anntype: str,
                 dimension: int,
                 order: list[int],
                 paras: list[int],
                 matrix: list,
                 scale: float,
                 scale_vis: float,
                 video_path: str):

        self._name = name
        self._anntype = anntype
        self._annpath = annpath
        self._dimension = dimension
        self._order = order
        self._paras = paras
        self._matrix = matrix
        self._scale = scale
        self._scale_vis = scale_vis
        self._video_path = video_path

    @staticmethod
    def get(dataset: str, clip: str, root_dir='./dataset_configs'):
        """
        Get a `VideoClip` object

        :param dataset: name of the image dataset
        :param clip: name of the video clip
        :param root_dir: dataset config folder
        """

        plist_path = os.path.join(root_dir,
                                  dataset,
                                  'subsets',
                                  '{}.plist'.format(clip))
        try:
            dic = load_from_plist(plist_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}` NOT FOUND.'.format(plist_path))

        return VideoClip(**dic)

    @property
    def name(self):
        """
        Name of the video clip.
        """
        return self._name

    @property
    def annpath(self) -> str:
        """
        Path of the annotation file. 
        """
        return self._annpath

    @property
    def anntype(self):
        """
        Type of annotations in this video clip.
        canbe `'coordinate'`, `'boundingbox'`, ...
        """
        return self._anntype

    @property
    def order(self) -> list[int]:
        """
        X-Y order in the annotation file.
        """
        return self._order

    @property
    def paras(self) -> tuple[int, int]:
        """
        [sample_step, frame_rate]
        """
        return self._paras

    @property
    def video_path(self) -> str:
        """
        Path of the video file.
        """
        return self._video_path

    @property
    def matrix(self) -> list[float]:
        """
        transfer weights from real scales to pixels.
        """
        return self._matrix

    @property
    def scale(self):
        """
        annotation scales
        """
        return self._scale

    @property
    def scale_vis(self):
        """
        scale when saving visualized images
        """
        return self._scale_vis

    @property
    def dimension(self):
        """
        Dimension of trajectories of each agent at each 
        time step in the dataset.
        For example, `[x, y]` -> `dimension = 2`.
        """
        return self._dimension


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

        self.info: VideoClip = VideoClip.get(self.dataset, name)
        self.custom_list = custom_list
        self.agent_count = None
        self.trajectories: list[Trajectory] = None

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
        - `item[2:M]`: dataset records, like coordinates, 
            bounding boxes, and other type of trajectory series.

        :param file_name: name of the annatation file
        """

        data = np.genfromtxt(file_name, dtype=np.str, delimiter=',')

        agents = {}
        agent_ids = np.unique(agent_order := data.T[1])

        try:
            _agent_ids = agent_ids.astype(np.int)
            agent_ids = np.sort(_agent_ids).astype(str)
        except:
            pass

        for id in agent_ids:
            index = np.where(agent_order == id)[0]
            agents[id] = np.delete(data[index], 1, axis=1)

        frame_ids = list(set(data.T[0].astype(np.int32)))
        frame_ids.sort()

        self.log('Dataset file {} loaded.'.format(file_name))
        return agents, frame_ids

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
            person_ids = dat['person_ids']

        # or start processing and then saving
        else:
            persons_appear, frame_ids = self.load_dataset(self.info.annpath)
            person_ids = list(persons_appear.keys())

            p = len(person_ids)
            f = len(frame_ids)

            # person_id -> person_index
            person_dict = dict(zip(person_ids, np.arange(p)))

            # frame_id -> frame_index
            frame_dict = dict(zip(frame_ids, np.arange(f)))

            # init the matrix
            dim = persons_appear[person_ids[0]].shape[-1] - 1
            matrix = INIT_POSITION * np.ones([f, p, dim])

            timebar = self.log_timebar(inputs=person_dict.items(),
                                       text='Processing dataset...',
                                       return_enumerate=False)
            for person_id, person_index in timebar:
                frame_id = persons_appear[person_id].T[0].astype(np.int32)
                frame_index = [frame_dict[fi] for fi in frame_id]

                matrix[frame_index, person_index, :] \
                    = persons_appear[person_id][:, 1:]

            neighbor_indexes = np.array([
                np.where(np.not_equal(data, INIT_POSITION))[0]
                for data in matrix[:, :, 0]], dtype=object)

            np.savez(npy_path,
                     neighbor_indexes=neighbor_indexes,
                     matrix=matrix,
                     frame_ids=frame_ids,
                     person_ids=person_ids)

        self.agent_count = matrix.shape[1]
        self.frame_number = matrix.shape[0]

        return neighbor_indexes, matrix, frame_ids, person_ids

    def make_trajectories(self):
        """
        Make trajectories from the processed dataset files.
        """
        if len(self.custom_list) == 4:
            self.nei_indexes, self.matrix, self.frame_ids, self.person_ids = self.custom_list
        else:
            self.nei_indexes, self.matrix, self.frame_ids, self.person_ids = self.process_metadata()

        trajs = []
        for person_index in range(self.agent_count):
            trajs.append(Trajectory(agent_id=self.person_ids[person_index],
                                    trajectory=self.matrix[:, person_index, :],
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

        timebar = self.log_timebar(range(self.agent_count),
                                   text='Prepare train data...')
        for agent_index, _ in timebar:
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
        for agent in self.log_timebar(agents,
                                      'Build maps...',
                                      return_enumerate=False):

            centers.append(agent.traj[-1:, :])
            social_maps.append(map_manager.build_social_map(
                target_agent=agent,
                traj_neighbors=agent.get_pred_traj_neighbor_linear()))

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
