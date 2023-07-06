"""
@Author: Conghao Wong
@Date: 2023-06-12 15:11:35
@LastEditors: Conghao Wong
@LastEditTime: 2023-07-06 10:33:53
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import copy

import numpy as np

from ...base import BaseManager
from ...utils import get_loss_mask
from .__picker import AnnotationManager


class BaseInputObject():
    """
    BaseInputObject
    ---

    The basic class to load dataset files directly.
    """

    __version__ = 0.0
    _save_items = []

    def __init__(self) -> None:

        self._id: str = None
        self._type: str = None

        self._traj: np.ndarray = None
        self._traj_future: np.ndarray = None

        self._traj_pred: np.ndarray = None
        self._traj_linear: np.ndarray = None

        self._frames: np.ndarray = None
        self._frames_future: np.ndarray = None

        self.obs_length = 0
        self.total_frame = 0

        self._mask: np.ndarray = None
        self.linear_predict = False

        self.manager: BaseManager = None

    def copy(self):
        return copy.deepcopy(self)

    @property
    def pickers(self) -> AnnotationManager:
        return self.manager.pickers

    def init_data(self):
        raise NotImplementedError

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

    ##########################################
    # Trajectories data
    # (observations, labels, predictions)
    ##########################################
    @property
    def traj(self):
        raise NotImplementedError

    @property
    def masked_traj(self):
        """
        Masked observed trajectories.
        """
        return self._get_masked_traj(self.traj)

    @property
    def groundtruth(self):
        raise NotImplementedError

    @property
    def masked_groundtruth(self):
        """
        Masked groundtruth future trajectories.
        """
        return self._get_masked_traj(self.groundtruth)

    @property
    def pred(self):
        raise NotImplementedError

    @property
    def masked_pred(self):
        """
        Masked future predicted trajectories.
        """
        return self._traj_pred

    @property
    def pred_linear(self):
        raise NotImplementedError

    def write_pred(self, pred: np.ndarray):
        raise NotImplementedError

    @property
    def mask(self) -> np.ndarray:
        """
        The mask matrix to show whether the trajectory is valid.
        """
        if self._mask is None:
            self._mask = get_loss_mask(self.traj, self.groundtruth,
                                       return_numpy=True)
        return self._mask

    ##########################################
    # Agent data
    # (frames, id)
    ##########################################
    @property
    def id(self) -> str:
        """
        Agent ID
        """
        return self._id

    @property
    def type(self) -> str:
        """
        Agent type
        """
        return self._type

    @property
    def frames(self) -> np.ndarray:
        """
        a list of frame indexes during observation and prediction time.
        shape = (obs + pred)
        """
        return np.concatenate([self._frames, self._frames_future])

    @property
    def frames_future(self) -> np.ndarray:
        """
        a list of frame indexes during prediction time.
        shape = (pred)
        """
        return self._frames_future

    def _get_masked_traj(self, traj: np.ndarray):
        if not issubclass(type(self.mask), np.ndarray):
            return traj
        else:
            index = np.where(self.mask)[0]
            return traj[index]

    def init_data(self, *args, **kwargs):
        raise NotImplementedError
