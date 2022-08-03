"""
@Author: Conghao Wong
@Date: 2022-06-21 09:26:56
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-03 15:38:54
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import copy

import numpy as np

from ..utils import INIT_POSITION


class Agent():
    """
    Agent
    -----
    Structure to manage data of one training sample.

    Properties
    ----------
    ```python
    self.traj -> np.ndarray     # historical trajectory
    self.pred -> np.ndarray     # predicted (future) trajectory
    self.frames -> list[int]    # a list of frame index when this agent appeared
    self.frames_future -> list[int]     # agent's future frame index
    self.pred_linear -> np.ndarray  # agent's linear prediction
    self.groundtruth -> np.ndarray  # agent's future trajectory (when available)

    self.Map  -> np.ndarray   # agent's context map
    ```

    Public Methods
    --------------
    ```python
    # copy this manager to a new address
    >>> self.copy() -> BasePredictionAgent

    # get neighbors' trajs -> list[np.ndarray]
    >>> self.get_neighbor_traj()

    # get neighbors' linear predictions
    >>> self.get_pred_traj_neighbor_linear() -> list[np.ndarray]
    ```
    """

    __version__ = 5.0

    _save_items = ['_traj', '_traj_future',
                   '_traj_pred', '_traj_pred_linear',
                   '_frames', '_frames_future',
                   '_id',
                   'real2grid', '__version__',
                   'linear_predict',
                   'neighbor_number',
                   'neighbor_traj',
                   'neighbor_traj_linear_pred',
                   'obs_length', 'total_frame']

    def __init__(self):

        self._traj: np.ndarray = None
        self._traj_future: np.ndarray = None

        self._traj_pred: np.ndarray = None
        self._traj_pred_linear: np.ndarray = None

        self._map = None
        self.real2grid = None

        self._id = None

        self._frames = []
        self._frames_future = []

        self.linear_predict = False
        self.obs_length = 0
        self.total_frame = 0

        self.neighbor_number = 0
        self.neighbor_traj: list[np.ndarray] = []
        self.neighbor_traj_linear_pred: list[np.ndarray] = []

        self.dim = None

    def copy(self):
        return copy.deepcopy(self)

    def get_ndim_trajectory(self, traj: np.ndarray, dim: int):
        """
        get the n-dim trajectory from the input.
        """
        true_dim = traj.shape[-1]
        dims = (dim, true_dim)

        if dims == (2, 4):
            xl, yl, xr, yr = traj.T
            return 0.5 * np.column_stack((xl+xr, yl+yr))

        elif dims == (2, 2):
            return traj

        elif dims == (4, 4):
            return traj

        else:
            raise NotImplementedError(dims)

    @property
    def id(self) -> str:
        """
        Agent ID
        """
        return self._id

    @property
    def traj(self) -> np.ndarray:
        """
        historical trajectory, shape = (obs, dim)
        """
        return self.get_ndim_trajectory(self._traj, self.dim)

    @property
    def pred(self) -> np.ndarray:
        """
        predicted trajectory, shape = (pred, dim)
        """
        return self._traj_pred

    @property
    def frames(self) -> list:
        """
        a list of frame index during observation and prediction time.
        shape = (obs + pred, 2)
        """
        return self._frames + self._frames_future

    @property
    def frames_future(self) -> list:
        """
        a list of frame index during prediction time.
        shape = (pred, 2)
        """
        return self._frames_future

    @property
    def pred_linear(self) -> np.ndarray:
        """
        linear prediction.
        shape = (pred, 2)
        """
        return self.get_ndim_trajectory(self._traj_pred_linear, self.dim)

    @property
    def groundtruth(self) -> np.ndarray:
        """
        ground truth future trajectory.
        shape = (pred, 2)
        """
        return self.get_ndim_trajectory(self._traj_future, self.dim)

    @property
    def Map(self) -> np.ndarray:
        """
        context map
        """
        return self._map

    def set_map(self, Map: np.ndarray, paras: np.ndarray):
        self._map = Map
        self.real2grid = paras

    def zip_data(self) -> dict[str, object]:
        zipped = {}
        for item in self._save_items:
            zipped[item] = getattr(self, item)
        return zipped

    def load_data(self, zipped_data: dict[str, object]):
        for item in self._save_items:
            if not item in zipped_data.keys():
                continue
            else:
                setattr(self, item, zipped_data[item])
        return self

    def init_data(self, id: str,
                  target_traj: np.ndarray,
                  neighbors_traj: np.ndarray,
                  frames: list[int],
                  start_frame, obs_frame, end_frame,
                  frame_step=1,
                  add_noise=False,
                  linear_predict=True):
        """
        Make one training data.

        NOTE that `start_frame`, `obs_frame`, `end_frame` are
        indexes of frames, not their ids.
        Length (time steps) of `target_traj` and `neighbors_traj`
        are `(end_frame - start_frame) // frame_step`.
        """

        self.linear_predict = linear_predict

        # Trajectory info
        self.obs_length = (obs_frame - start_frame) // frame_step
        self.total_frame = (end_frame - start_frame) // frame_step

        # data strengthen: noise
        if add_noise:
            target_traj += np.random.normal(0, 0.1, target_traj.shape)

        self._id = id
        self._frames = frames[:self.obs_length]
        self._traj = target_traj[:self.obs_length]
        self._traj_future = target_traj[self.obs_length:]
        self._frames_future = frames[self.obs_length:]

        # Neighbor info
        self.clear_all_neighbor_info()
        for _n_traj in neighbors_traj:
            if _n_traj.max() >= INIT_POSITION:
                index = np.where(_n_traj.T[0] < INIT_POSITION)[0]
                _n_traj[:index[0], :] = _n_traj[index[0]]
                _n_traj[index[-1]:, :] = _n_traj[index[-1]]

            self.neighbor_traj.append(_n_traj)

        self.neighbor_number = len(neighbors_traj)

        if linear_predict:
            pred_frames = self.total_frame - self.obs_length
            n = self.neighbor_number

            self._traj_pred_linear = linear_pred(self._traj,
                                                 self.obs_length,
                                                 pred_frames)

            _n_pred = linear_pred(np.concatenate(self.neighbor_traj, axis=-1),
                                  self.obs_length,
                                  pred_frames)

            _n_pred = np.reshape(_n_pred, [pred_frames, n, -1])
            _n_pred = np.transpose(_n_pred, [1, 0, 2])
            self.neighbor_traj_linear_pred = _n_pred

        return self

    def get_neighbor_traj(self):
        return self.neighbor_traj

    def clear_all_neighbor_info(self):
        self.neighbor_traj = []
        self.neighbor_traj_linear_pred = []

    def get_pred_traj_neighbor_linear(self) -> list:
        return self.neighbor_traj_linear_pred


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def __predict_linear(x, y, x_p, P):
    """
    Linear prediction

    :param x: shape = (batch, obs)
    :param y: shape = (batch, pred)
    :param P: shape = (obs, obs)
    """

    A = np.stack([np.ones_like(x), x]).T        # (obs, 2)
    A_p = np.stack([np.ones_like(x_p), x_p]).T  # (pred, 2)
    Y = y.T  # (obs)
    B = np.linalg.inv(A.T @ P @ A) @ A.T @ P @ Y
    Y_p = A_p @ B
    return Y_p


def linear_pred(inputs, obs_frames, pred_frames,
                diff_weights=0.95) -> np.ndarray:

    if diff_weights == 0:
        P = np.diag(np.ones(shape=[obs_frames]))
    else:
        P = np.diag(softmax([(i+1)**diff_weights for i in range(obs_frames)]))

    t = np.arange(obs_frames)
    t_p = np.arange(obs_frames + pred_frames)
    dim = inputs.shape[-1]

    inputs = np.transpose(inputs, [1, 0])
    pred = __predict_linear(t, inputs, t_p, P)
    return pred[obs_frames:]
