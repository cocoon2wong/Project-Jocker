"""
@Author: Conghao Wong
@Date: 2022-06-21 09:41:10
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-05 15:00:30
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import plistlib


class VideoClip():
    """
    VideoClip
    ---------
    Base structure for controlling each video dataset.

    Properties
    -----------------
    ```python
    >>> self.name           # dataset name
    >>> self.dataset_dir    # dataset folder
    >>> self.paras          # [sample_step, frame_rate]
    >>> self.video_path     # video path
    >>> self.weights        # transfer weights from real scales to pixels
    >>> self.scale          # video scales
    ```
    """

    def __init__(self, name: str,
                 dataset_dir: str,
                 paras: list[int],
                 video_path: str,
                 weights: list,
                 scale: float,
                 dimension: int = 2,
                 anntype: str = None):

        self._name = name
        self._dataset_dir = dataset_dir
        self._paras = paras
        self._video_path = video_path
        self._weights = weights
        self._scale = scale
        self._dimension = dimension
        self._anntype = anntype

    @staticmethod
    def get(dataset: str, root_dir='./datasets/subsets'):
        plist_path = os.path.join(root_dir, '{}.plist'.format(dataset))
        try:
            dic = load_from_plist(plist_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}`.plist NOT FOUND.'.format(dataset))

        return VideoClip(**dic)

    @property
    def name(self):
        """
        Name of the video clip.
        """
        return self._name

    @property
    def dataset_dir(self):
        """
        Dataset folder, which contains a `*.txt` or `*.csv` 
        dataset file, and a scene image `reference.jpg`.
        """
        return self._dataset_dir

    @property
    def paras(self):
        """
        [sample_step, frame_rate]
        """
        return self._paras

    @property
    def video_path(self):
        return self._video_path

    @property
    def weights(self):
        return self._weights

    @property
    def scale(self):
        return self._scale

    @property
    def dimension(self):
        """
        Dimension of trajectories of each agent at each 
        time step in the dataset.
        For example, `[x, y]` -> `dimension = 2`.
        """
        return self._dimension

    @property
    def anntype(self):
        """
        Type of annotations in this video clip.
        canbe `'coordinate'`, `'boundingbox'`, ...
        """
        return self._anntype


class Dataset():
    """
    Dataset
    -------
    Manage a full trajectory prediction dataset.
    A dataset may contains several video clips.

    """

    def __init__(self, name: str, root_dir='./datasets'):
        """
        init

        :param name: name of the dataset, NOT the name of the video clips
        :param root_dir: path where the dataset `plist` file puts
        """
        plist_path = os.path.join(root_dir, '{}.plist'.format(name))
        try:
            dic = load_from_plist(plist_path)
        except:
            raise FileNotFoundError(
                'Dataset file `{}`.plist NOT FOUND.'.format(name))

        self.train_sets: list[str] = dic['train']
        self.test_sets: list[str] = dic['test']
        self.val_sets: list[str] = dic['val']

        self._dimension: int = dic['dimension']
        self._weights: float = dic['weights']
        self._anntype: str = dic['anntype']

    @property
    def dimension(self) -> int:
        """
        Maximum dimension of trajectories recorded in this dataset.
        For example, `(x, y)` -> `dimension = 2`.
        """
        return self._dimension

    @property
    def weights(self) -> float:
        """
        Maximum pixel length of the images.
        For example, `weights = 1920` when `(H, W) = (1920, 1080)`.
        """
        return self._weights

def load_from_plist(path: str) -> dict:
    """
    Load plist files into python `dict` object.

    :param path: path of the plist file
    :return dat: a `dict` object loaded from the file
    """
    with open(path, 'rb') as f:
        dat = plistlib.load(f)

    return dat
