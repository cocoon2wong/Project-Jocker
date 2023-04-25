"""
@Author: Conghao Wong
@Date: 2022-08-30 09:52:17
@LastEditors: Conghao Wong
@LastEditTime: 2023-04-25 11:01:34
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import numpy as np
import tensorflow as tf

from ...base import BaseManager
from ...constant import ANN_TYPES


class _BaseAnnType():
    def __init__(self) -> None:
        self.typeName: str = None
        self.dim: int = None
        self.targets: list[type[_BaseAnnType]] = []

    def transfer(self, target, traj: np.ndarray) -> np.ndarray:
        """
        Transfer the n-dim trajectory to the other m-dim trajectory.

        :param target: An instance or subclass of `_BaseAnnType` that \
            manages the m-dim trajectory.
        :param traj: The n-dim trajectory.
        """
        T = type(target)
        if T == type(self):
            return traj

        if not T in self.targets:
            T_c = self.__class__.__name__
            raise ValueError(f'Transfer from {T_c} to {T} is not supported.')

        else:
            return self._transfer(T, traj)

    def _transfer(self, target, traj: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _SeriesOfSingleCoordinate(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = ANN_TYPES._CO_SERIES_2D
        self.dim = None
        self.targets = []


class _Coordinate(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = ANN_TYPES.CO_2D
        self.dim = 2
        self.targets = [_SeriesOfSingleCoordinate]

    def _transfer(self, T: type[_BaseAnnType], traj: np.ndarray):
        if T == _SeriesOfSingleCoordinate:
            # return shape = (1, ..., steps, 2)
            return [traj]

        else:
            raise NotImplementedError(T)


class _Boundingbox(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = ANN_TYPES.BB_2D
        self.dim = 4
        self.targets = [_Coordinate, _SeriesOfSingleCoordinate]

    def _transfer(self, T: type[_BaseAnnType], traj: np.ndarray):

        p1 = traj[..., 0:2]
        p2 = traj[..., 2:4]

        if T == _Coordinate:
            # return shape = (..., steps, 2)
            return 0.5 * (p1 + p2)

        elif T == _SeriesOfSingleCoordinate:
            # return shape = (2, ..., steps, 2)
            return [p1, p2]

        else:
            raise NotImplementedError(T)


class _3DBoundingbox(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = ANN_TYPES.BB_3D
        self.dim = 6
        self.targets = [_Coordinate,
                        _SeriesOfSingleCoordinate]

    def _transfer(self, T: type[_BaseAnnType], traj: np.ndarray):

        p1 = traj[..., 0:3]
        p2 = traj[..., 3:6]

        if T == _Coordinate:
            # return shape = (..., steps, 2)
            return 0.5 * (p1 + p2)[..., 0:2]

        elif T == _SeriesOfSingleCoordinate:
            # return shape = (2, ..., steps, 3)
            return [p1, p2]

        else:
            raise NotImplementedError


class _3DBoundingboxWithRotate(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = ANN_TYPES.BB_3D_R
        self.dim = 10
        self.targets = [_Coordinate,
                        _SeriesOfSingleCoordinate,
                        _3DBoundingbox]

    def _transfer(self, T: type[_BaseAnnType], traj: np.ndarray):

        p1 = traj[..., 0:3]
        p2 = traj[..., 3:6]

        if T == _Coordinate:
            # return shape = (..., steps, 2)
            return 0.5 * (p1 + p2)[..., 0:2]

        elif T == _SeriesOfSingleCoordinate:
            # return shape = (2, ..., steps, 3)
            return [p1, p2]

        elif T == _3DBoundingbox:
            # return shape = (..., steps, 6)
            return np.concatenate([p1, p2], axis=-1)

        else:
            raise NotImplementedError


class _3DSkeleton17(_BaseAnnType):
    def __init__(self) -> None:
        self.typeName = ANN_TYPES.SKE_3D_17
        self.dim = 17 * 3
        self.targets = [_Coordinate,
                        _SeriesOfSingleCoordinate]

    def _transfer(self, T: type[_BaseAnnType], traj: np.ndarray):
        source_shape = list(traj.shape)
        points = np.reshape(traj, source_shape[:-1] + [17, 3])

        if T == _Coordinate:
            # return shape = (..., steps, 2)
            return np.mean(points, axis=-2)[..., 0:2]

        elif T == _SeriesOfSingleCoordinate:
            # return shape = (17, ..., steps, 3)
            return np.array([points[..., i, :] for i in range(17)])

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
        Both arguments `datasetType` and `predictionType` accept strings:
        - `'coordinate'`
        - `'boundingbox'`
        - `'boundingbox-rotate'`
        - `'3Dboundingbox'`
        - `'3Dboundingbox-rotate'`
        - `'3Dskeleton-17'`

        :param datasetType: The type of the dataset annotation files.
        :param predictionType: The type of the model predictions.
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


class AnnotationManager(BaseManager):
    """
    Annotation Manager
    ---
    A manager to control all annotations and their transformations
    in dataset files and prediction models. The `AnnotationManager`
    object is managed by the `Structure` object directly.
    """

    def __init__(self, manager: BaseManager,
                 dataset_type: str,
                 name: str = 'Annotation Manager'):

        super().__init__(manager=manager, name=name)

        self.d_type = dataset_type
        self.p_type = self.args.anntype

        if self.args.force_anntype != 'null':
            self.p_type = self.args.force_anntype

        self.dataset_picker = Picker(datasetType=dataset_type,
                                     predictionType=self.p_type)

        self.center_picker = Picker(datasetType=self.p_type,
                                    predictionType=ANN_TYPES.CO_2D)

        self.single_picker = Picker(datasetType=self.p_type,
                                    predictionType=ANN_TYPES._CO_SERIES_2D)

    def get(self, inputs: Union[tf.Tensor, np.ndarray]):
        """
        Get data with target annotations from original dataset files.
        """
        return self.dataset_picker.get(inputs)

    def get_center(self, inputs: Union[tf.Tensor, np.ndarray]):
        """
        Get the center of trajectories from the processed data.
        Note that the annotation type of `inputs` is the same as the model's
        prediction type. (Not the dataset's annotation type.)
        """
        return self.center_picker.get(inputs)

    def get_coordinate_series(self, inputs: Union[tf.Tensor, np.ndarray]) \
            -> Union[list[tf.Tensor], list[np.ndarray]]:
        """
        Reshape inputs trajectories into a series of single coordinates.
        Note that the annotation type of `inputs` is the same as the model's
        prediction type. (Not the dataset's annotation type.)
        For example, when inputs have the annotation type `boundingbox`,
        then this function will reshape it into two slices of single
        2D coordinates with shapes `(..., steps, 2)`, and then return
        a list containing them.
        """
        return self.single_picker.get(inputs)

    def print_info(self, **kwargs):
        info = {'Dataset annotation type': self.d_type,
                'Model prediction type': self.p_type}

        kwargs.update(**info)
        return super().print_info(**kwargs)


def get_manager(anntype: str) -> _BaseAnnType:

    if anntype == ANN_TYPES.CO_2D:
        return _Coordinate()
    elif anntype == ANN_TYPES.BB_2D:
        return _Boundingbox()
    elif anntype == ANN_TYPES.BB_2D_R:
        raise NotImplementedError(anntype)

    elif anntype == ANN_TYPES.BB_3D:
        return _3DBoundingbox()
    elif anntype == ANN_TYPES.BB_3D_R:
        return _3DBoundingboxWithRotate()
    elif anntype == ANN_TYPES.SKE_3D_17:
        return _3DSkeleton17()

    elif anntype == ANN_TYPES._CO_SERIES_2D:
        return _SeriesOfSingleCoordinate()

    else:
        raise NotImplementedError(anntype)


def isNumpy(value):
    if issubclass(type(value), np.ndarray):
        return True
    else:
        return False


def isTensor(value):
    if issubclass(type(value), tf.Tensor):
        return True
    else:
        return False
