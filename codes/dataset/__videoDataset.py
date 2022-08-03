"""
@Author: Conghao Wong
@Date: 2022-07-19 11:19:58
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-03 10:33:21
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

from ..utils import DATASET_DIR, load_from_plist


class Dataset():
    """
    Dataset
    -------
    Manage a full trajectory prediction dataset.
    A dataset may contains several video clips.
    """

    # Saving paths
    BASE_DIR = DATASET_DIR
    CONFIG_FILE = os.path.join(BASE_DIR, '{}', '{}.plist')

    def __init__(self, name: str, split: str):
        """
        init

        :param name: name of the image dataset
        :param split: split name of the dataset
        :param root_dir: dataset config folder
        """
        split_path = self.CONFIG_FILE.format(name, split)

        try:
            dic = load_from_plist(split_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}` NOT FOUND.'.format(split_path))

        self.__name = dic['dataset']
        self.__type = dic['type']
        self.__scale = dic['scale']
        self.__scale_vis = dic['scale_vis']
        self.__dimension = dic['dimension']
        self.__anntype = dic['anntype']
        
        self.split: str = split
        self.train_sets: list[str] = dic['train']
        self.test_sets: list[str] = dic['test']
        self.val_sets: list[str] = dic['val']

    @property
    def name(self) -> str:
        """
        Name of the video dataset.
        For example, `ETH-UCY` or `SDD`.
        """
        return self.__name

    @property
    def type(self) -> str:
        """
        Annotation type of the dataset.
        For example, `'pixel'` or `'meter'`.
        """
        return self.__type

    @property
    def scale(self) -> float:
        """
        Global data scaling scale.
        """
        return self.__scale

    @property
    def scale_vis(self) -> float:
        """
        Video scaling when saving visualized results.
        """
        return self.__scale_vis

    @property
    def dimension(self) -> int:
        """
        Maximum dimension of trajectories recorded in this dataset.
        For example, `(x, y)` -> `dimension = 2`.
        """
        return self.__dimension

    @property
    def anntype(self) -> str:
        """
        Type of annotations.
        For example, `'coordinate'` or `'boundingbox'`.
        """
        return self.__anntype
