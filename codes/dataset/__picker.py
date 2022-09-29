"""
@Author: Conghao Wong
@Date: 2022-08-30 09:52:17
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-29 19:59:58
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np


class _BaseAnnType():
    def __init__(self) -> None:
        self.typeName: str = None
        self.dim: int = None
        self.order2d: list[list[int]] = None
        self.targets: list[type[_BaseAnnType]] = []

    def transfer(self, target, traj: np.ndarray) -> np.ndarray:
        """
        Transfer the n-dim trajectory to the other m-dim trajectory.

        :param target: an instance or subclass of `_BaseAnnType` that \
            manages the m-dim trajectory 
        :param traj: n-dim trajectory
        """
        T = type(target)
        if T == type(self):
            return traj

        if not T in self.targets:
            T_c = self.__class__.__name__
            raise ValueError(f'Transfer from {T_c} to {T} is not supported.')

        else:
            if traj.ndim == 4:       # (batch, steps, K, dim)
                _traj = np.transpose(traj, [3, 0, 1, 2])
            elif traj.ndim == 3:      # (batch, steps, dim)
                _traj = np.transpose(traj, [2, 0, 1])
            elif traj.ndim == 2:    # (steps, dim)
                _traj = traj.T
            else:
                raise NotImplementedError

            # shape of `_traj` is (dim, ...)
            return self._transfer(T, _traj)

    def _transfer(self, target, traj: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gather_dim(self, traj: np.ndarray, dim: int):
        return traj.T[dim].T

    def real2pixel2d(self, traj: np.ndarray, w, b, order=[0, 1]):
        """
        Transform n-dim trajectories into a series of 2d coordinates
        """
        results = []
        for _order in self.order2d:
            x = _order[order[0]]
            y = _order[order[1]]

            _x = self.gather_dim(traj, x)
            _y = self.gather_dim(traj, y)

            _xp = w[0] * _x + b[0]
            _yp = w[1] * _y + b[1]

            results += [_xp, _yp]

        return np.stack(results, axis=-1)


class _Coordinate(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = 'coordinate'
        self.dim = 2
        self.order2d = [[0, 1]]
        self.targets = []


class _Boundingbox(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = 'boundingbox'
        self.dim = 4
        self.order2d = [[0, 1], [2, 3]]
        self.targets = [_Coordinate]

    def _transfer(self, T: type[_BaseAnnType], traj: np.ndarray):
        if T == _Coordinate:
            xl, yl, xr, yr = traj[:4]
            return 0.5 * np.stack((xl+xr, yl+yr), axis=-1)

        else:
            raise NotImplementedError


class _3DBoundingboxWithRotate(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = '3Dboundingbox-rotate'
        self.dim = 10
        self.order2d = [[0, 1], [2, 3]]
        self.targets = [_Coordinate]

    def _transfer(self, T: type[_BaseAnnType], traj: np.ndarray):
        if T == _Coordinate:
            xl, yl, xr, yr = traj[:4]
            return 0.5 * np.stack((xl+xr, yl+yr), axis=-1)

        else:
            raise NotImplementedError


class Picker():
    """
    Picker
    ---

    Picker object to get trajectories from the n-dim meta-trajectories.
    """

    def __init__(self, datasetType: str, predictionType: str):
        """
        Both argument `datasetType` and `predictionType` accept strings:
        - `'coordinate'`
        - `'boundingbox'`
        - `'boundingbox-rotate'`
        - `'3Dboundingbox'`
        - `'3Dboundingbox-rotate'`

        :param datasetType: type of the dataset annotation files
        :param predictionType: type of the model predictions
        """
        super().__init__()

        self.ds_type = datasetType
        self.pred_type = predictionType

        self.ds_manager = get_manager(datasetType)
        self.pred_manager = get_manager(predictionType)

    def get(self, traj: np.ndarray) -> np.ndarray:
        """
        Get trajectories from the n-dim meta-trajectories.
        """
        return self.ds_manager.transfer(self.pred_manager, traj)

    def real2pixel2d(self, traj: np.ndarray, w, b, order=[0, 1]):
        """
        Transform n-dim trajectories into a series of 2d coordinates
        """
        return self.ds_manager.real2pixel2d(traj, w, b, order)


def get_manager(anntype: str) -> _BaseAnnType:
    if anntype == 'coordinate':
        return _Coordinate()
    elif anntype == 'boundingbox':
        return _Boundingbox()
    elif anntype == 'boundingbox-rotate':
        raise NotImplementedError(anntype)
    elif anntype == '3Dboundingbox':
        raise NotImplementedError(anntype)
    elif anntype == '3Dboundingbox-rotate':
        return _3DBoundingboxWithRotate()
    else:
        raise NotImplementedError(anntype)
