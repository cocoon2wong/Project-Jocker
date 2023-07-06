"""
@Author: Conghao Wong
@Date: 2023-06-12 10:16:03
@LastEditors: Conghao Wong
@LastEditTime: 2023-07-06 10:35:24
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from ...basemodels.layers import LinearLayerND
from ..__base import BaseInputObject

LINEAR_LAYER = None


class Frame(BaseInputObject):
    """
    Frame
    ---
    Manage all agents from a frame in the video.

    Properties
    ---

    """

    __version__ = 0.1
    _save_items = ['__version__',
                   '_traj', '_traj_future',
                   '_traj_pred', '_traj_linear',
                   '_init_position',
                   '_id', '_agent_ids', '_agent_types',
                   '_frames', '_frames_future',
                   'linear_predict',
                   'obs_length', 'total_frame']

    def __init__(self):

        super().__init__()

        self._init_position: float = None

        self._type = 'Frame'

        self._agent_ids: np.ndarray = None
        self._agent_types: np.ndarray = None

    def padding(self, trajs: np.ndarray) -> np.ndarray:
        """
        Padding all agents' trajectories.
        Shape should be `(n_agent, steps, dim)`.
        """
        n = len(trajs)
        m = self.manager.args.max_agents

        if n <= m:
            zero_pad = np.pad(trajs,
                              ((0, m-n), (0, 0), (0, 0)))
            zero_pad[n:, :, :] = self._init_position
        else:
            zero_pad = trajs[:m]

        return zero_pad

    @property
    def traj(self) -> np.ndarray:
        """
        Trajectory matrix of all observed frames.
        Shape is `(n_agent, obs, dim)`.
        The position of agents that are not in the scene will be
        represented as a big float number (`init_position` in this
        object).
        """
        return self.padding(self.pickers.get(self._traj))

    @property
    def agent_ids(self) -> np.ndarray:
        return self._agent_ids

    @property
    def agent_types(self) -> np.ndarray:
        return self._agent_types

    @property
    def groundtruth(self) -> np.ndarray:
        """
        ground truth future trajectory.
        shape = (n_agent, pred, dim)
        """
        return self.padding(self.pickers.get(self._traj_future))

    @property
    def pred(self) -> np.ndarray:
        """
        predicted trajectory, shape = (n_agent, pred, dim)
        """
        return self._traj_pred

    @property
    def pred_linear(self) -> np.ndarray:
        """
        linear prediction.
        shape = (n_agent, pred, dim)
        """
        return self.padding(self.pickers.get(self._traj_linear))

    def write_pred(self, pred: np.ndarray):
        self._traj_pred = self._get_masked_traj(pred)

    def init_data(self, id: str,
                  traj: np.ndarray,
                  frames: np.ndarray,
                  agent_ids: np.ndarray,
                  agent_types: np.ndarray,
                  start_frame: int,
                  obs_frame: int,
                  end_frame: int,
                  init_position: float,
                  frame_step: int = 1,
                  linear_predict=True):
        """
        Make one training data.

        NOTE that `start_frame`, `obs_frame`, `end_frame` are
        indices of frames, not their ids.
        Length (time steps) of `target_traj` and `neighbors_traj`
        are `(end_frame - start_frame) // frame_step`.
        """

        self.linear_predict = linear_predict
        self._init_position = init_position

        # Trajectory info
        self.obs_length = (obs_frame - start_frame) // frame_step
        self.total_frame = (end_frame - start_frame) // frame_step

        self._id = id
        self._agent_ids = agent_ids
        self._agent_types = agent_types
        self._traj = traj[..., :self.obs_length, :]
        self._traj_future = traj[..., self.obs_length:, :]

        frames = np.array(frames)
        self._frames = frames[:self.obs_length]
        self._frames_future = frames[self.obs_length:]

        if linear_predict:
            pred_frames = self.total_frame - self.obs_length

            global LINEAR_LAYER
            if not LINEAR_LAYER:
                LINEAR_LAYER = LinearLayerND(obs_frames=self.obs_length,
                                             pred_frames=pred_frames)

            self._traj_linear = LINEAR_LAYER(self._traj).numpy()

        return self
