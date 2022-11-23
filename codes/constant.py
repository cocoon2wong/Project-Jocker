"""
@Author: Conghao Wong
@Date: 2022-11-23 18:01:16
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-23 18:29:01
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""


class INPUT_TYPES():
    """
    Type names of all kinds of model inputs.
    """
    OBSERVED_TRAJ = 'TRAJ'
    MAP = 'MAP'
    DESTINATION_TRAJ = 'DEST'
    GROUNDTRUTH_TRAJ = 'GT'
    GROUNDTRUTH_SPECTRUM = 'GT_SPECTRUM'
    ALL_SPECTRUM = 'ALL_SPECTRUM'


class PROCESS_TYPES():
    """
    Names of all pre-process and post-process methods.
    """
    MOVE = 'MOVE'
    ROTATE = 'ROTATE'
    SCALE = 'SCALE'
    UPSAMPLING = 'UPSAMPLING'
