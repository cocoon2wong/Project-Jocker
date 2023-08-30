"""
@Author: Conghao Wong
@Date: 2022-11-23 18:01:16
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-30 17:00:45
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""


class ANN_TYPES():
    """
    Type names of heterogeneous trajectories.
    """
    CO_2D = 'coordinate'
    CO_3D = '3Dcoordinate'
    BB_2D = 'boundingbox'
    BB_2D_R = 'boundingbox-rotate'
    BB_3D = '3Dboundingbox'
    BB_3D_R = '3Dboundingbox-rotate'
    SKE_3D_17 = '3Dskeleton-17'

    _CO_SERIES_2D = 'coordinate-series'


class INPUT_TYPES():
    """
    Type names of all kinds of model inputs.
    """
    OBSERVED_TRAJ = 'TRAJ'
    NEIGHBOR_TRAJ = 'TRAJ_NEIGHBOR'
    MAP = 'MAP'
    MAP_PARAS = 'MAP_PARAS'
    DESTINATION_TRAJ = 'DEST'
    GROUNDTRUTH_TRAJ = 'GT'
    GROUNDTRUTH_SPECTRUM = 'GT_SPECTRUM'
    ALL_SPECTRUM = 'ALL_SPECTRUM'


class OUTPUT_TYPES():
    """
    Type names of all kinds of model outputs.
    """
    PREDICTED_TRAJ = 'TRAJ_PREDICTED'


class PROCESS_TYPES():
    """
    Names of all pre-process and post-process methods.
    """
    MOVE = 'MOVE'
    ROTATE = 'ROTATE'
    SCALE = 'SCALE'
    UPSAMPLING = 'UPSAMPLING'


class INTERPOLATION_TYPES():
    """
    Names of all interpolation methods.
    """

    LINEAR = 'l'
    LINEAR_SPEED = 'speed'
    LINEAR_ACC = 'acc'
    NEWTON = 'newton'

    @classmethod
    def get_type(cls, s: str):
        for _s in [cls.LINEAR, cls.LINEAR_ACC,
                   cls.LINEAR_SPEED, cls.NEWTON]:
            if s.startswith(_s):
                return _s
        return None
