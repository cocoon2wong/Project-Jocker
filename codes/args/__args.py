"""
@Author: Conghao Wong
@Date: 2022-06-20 10:53:48
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-26 16:12:20
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import json
import os
import re
from typing import Any

from ..utils import DATASET_DIR, TIME, dir_check


class Args():
    """
    A set of args used for training or evaluating prediction models.
    """

    def __init__(self, terminal_args: list[str] = None) -> None:

        # args that load from the saved json file
        self._args_load: dict[str, Any] = {}

        # args that obtained from terminal
        self._args_runnning: dict[str, Any] = {}

        # args that set manually
        self._args_manually: dict[str, Any] = {}

        # default args
        self._args_default: dict[str, Any] = {}

        # a list that contains all args' names
        self._arg_list = [s for s in self.__dir__() if not s.startswith('_')]
        self._arg_list.sort()

        if terminal_args:
            self._load_from_terminal(terminal_args)

        if (l := self.load) != 'null':
            self._load_from_json(l)

    def _load_from_json(self, dir_path: str):
        try:
            arg_paths = [(p := os.path.join(dir_path, item)) for item in os.listdir(dir_path) if (
                item.endswith('args.json'))]

            with open(p, 'r') as f:
                json_dict = json.load(f)

            self._args_load = json_dict

        except:
            raise ValueError(f'Failed to load args from `{dir_path}`.')

        return self

    def _load_from_terminal(self, argv: list[str]):
        dic = {}

        index = 1
        while True:
            try:
                if argv[index].startswith('--'):
                    name = argv[index][2:]
                    value = argv[index+1]

                dic[name] = value
                index += 2

            except:
                break

        self._args_runnning = dic
        return self

    def _save_as_json(self, target_dir: str):
        dir_check(target_dir)
        json_path = os.path.join(target_dir, 'args.json')

        names = self._arg_list
        values = [getattr(self, s) for s in self._arg_list]

        with open(json_path, 'w+') as f:
            json.dump(dict(zip(names, values)), f,
                      separators=(',\n', ':'))

    def _get_args_by_index_and_name(self, index: int, name: str):
        if index == 0:
            dic = self._args_load
        elif index == 1:
            dic = self._args_runnning
        elif index == 99:
            dic = self._args_manually
        elif index == -1:
            dic = self._args_default
        else:
            raise ValueError('Args index not exist.')

        return dic[name] if name in dic.keys() else None

    def _set(self, name: str, value: Any):
        """
        Set argument manually.
        """
        self._args_manually[name] = value

    def _set_default(self, name: str, value: Any):
        """
        Set default argument values.
        """
        self._args_default[name] = value

    def _get(self, name: str, default: Any, argtype: str):
        """
        Get arg by name

        :param name: name of the arg
        :param default: default value of the arg
        :param argtype: type of the arg, canbe
            - `'static'`
            - `'dynamic'`
            - ...
        """

        # arg dict index:
        # _args_load: 0
        # _args_running: 1
        # _args_manually: 99
        # _args_default: -1

        if argtype == 'static':
            order = [99, 0, 1, -1]
        elif argtype == 'dynamic':
            order = [99, 1, 0, -1]
        else:
            raise ValueError('Wrong arg type.')

        value = None
        for index in order:
            value = self._get_args_by_index_and_name(index, name)

            if value is not None:
                break
            else:
                continue

        if value is None:
            value = default

        value = type(default)(value)

        return value

    """
    Basic Model Args
    """
    @property
    def batch_size(self) -> int:
        """
        Batch size when implementation.
        """
        return self._get('batch_size', 5000, argtype='dynamic')

    @property
    def dataset(self) -> str:
        """
        Name of the video dataset to train or evaluate.
        For example, `'ETH-UCY'` or `'SDD'`.
        DO NOT set this argument manually.
        """
        if self.force_dataset != 'null':
            return self.force_dataset

        if not 'dataset' in self._args_default.keys():
            dirs = os.listdir(DATASET_DIR)

            plist_files = []
            for d in dirs:
                try:
                    _path = os.path.join(DATASET_DIR, d)
                    for p in os.listdir(_path):
                        if p.endswith('.plist'):
                            plist_files.append(os.path.join(_path, p))
                except:
                    pass

            dataset = None
            for f in plist_files:
                res = re.findall(f'{DATASET_DIR}/(.*)/({self.split}.plist)', f)

                if len(res):
                    dataset = res[0][0]
                    break

            if not dataset:
                raise ValueError(self.split)

            self._set_default('dataset', dataset)

        return self._get('dataset', 'error', argtype='static')

    @property
    def epochs(self) -> int:
        """
        Maximum training epochs.
        """
        return self._get('epochs', 500, argtype='static')

    @property
    def force_clip(self) -> str:
        """
        Force test video clip.
        It only works when `test_mode` is `one`. 
        """
        if not self.draw_results in ['null', '0', '1']:
            self._set('force_clip', self.draw_results)

        return self._get('force_clip', 'null', argtype='dynamic')

    @property
    def force_dataset(self) -> str:
        """
        Force test dataset. 
        """
        return self._get('force_dataset', 'null', argtype='dynamic')

    @property
    def force_split(self) -> str:
        """
        Force test dataset. 
        Only works when evaluating when `test_mode` is `one`.
        """
        return self._get('force_split', 'null', argtype='dynamic')

    @property
    def gpu(self) -> str:
        """
        Speed up training or test if you have at least one nvidia GPU. 
        If you have no GPUs or want to run the code on your CPU, 
        please set it to `-1`.
        """
        return self._get('gpu', '0', argtype='dynamic')

    @property
    def save_base_dir(self) -> str:
        """
        Base folder to save all running logs.
        """
        return self._get('save_base_dir', './logs', argtype='static')

    @property
    def save_model(self) -> int:
        """
        Controls if save the final model at the end of training.
        """
        return self._get('save_model', 1, argtype='static')

    @property
    def start_test_percent(self) -> float:
        """
        Set when to start validation during training.
        Range of this arg is `0 <= x <= 1`. 
        Validation will start at `epoch = args.epochs * args.start_test_percent`.
        """
        return self._get('start_test_percent', 0.0, argtype='static')

    @property
    def log_dir(self) -> str:
        """
        Folder to save training logs and models. If set to `null`,
        logs will save at `args.save_base_dir/current_model`.
        """
        if not 'log_dir' in self._args_default.keys():
            log_dir_current = (TIME +
                               self.model_name +
                               self.model +
                               self.split)
            default_log_dir = os.path.join(dir_check(self.save_base_dir),
                                           log_dir_current)

            self._set_default('log_dir', dir_check(default_log_dir))

        return self._get('log_dir', 'null', argtype='static')

    @property
    def load(self) -> str:
        """
        Folder to load model. If set to `null`,
        it will start training new models according to other args.
        """
        return self._get('load', 'null', argtype='dynamic')

    @property
    def model(self) -> str:
        """
        Model type used to train or test.
        """
        return self._get('model', 'none', argtype='static')

    @property
    def model_name(self) -> str:
        """
        Customized model name.
        """
        return self._get('model_name', 'model', argtype='static')

    @property
    def restore(self) -> str:
        """
        Path to restore the pre-trained weights before training.
        It will not restore any weights if `args.restore == 'null'`.
        """
        return self._get('restore', 'null', argtype='dynamic')

    @property
    def split(self) -> str:
        """
        Dataset split used when training and evaluating.
        """
        if self.force_split != 'null':
            return self.force_split

        if 'test_set' in self._args_load.keys():
            return self._get('test_set', 'zara1', argtype='static')

        return self._get('split', 'zara1', argtype='static')

    @property
    def test_step(self) -> int:
        """
        Epoch interval to run validation during training.
        """
        return self._get('test_step', 3, argtype='static')

    """
    Trajectory Prediction Args
    """
    @property
    def obs_frames(self) -> int:
        """
        Observation frames for prediction.
        """
        return self._get('obs_frames', 8, argtype='static')

    @property
    def pred_frames(self) -> int:
        """
        Prediction frames.
        """
        return self._get('pred_frames', 12, argtype='static')

    @property
    def draw_results(self) -> str:
        """
        Controls if draw visualized results on video frames.
        Accept the name of one video clip.
        The codes will first try to load the video according to the path
        saved in the `plist` file, and if successful it will draw the
        visualization on the video, otherwise it will draw on a blank canvas.
        Note that `test_mode` will be set to `'one'` and `force_split`
        will be set to `draw_results` if `draw_results != 'null'`.
        """
        return self._get('draw_results', 'null', argtype='dynamic')

    @property
    def draw_distribution(self) -> int:
        """
        Conrtols if draw distributions of predictions instead of points.
        If `draw_distribution == 0`, it will draw results as normal coordinates;
        If `draw_distribution == 1`, it will draw results from all future time
        steps together in the distribution way;
        If `draw_distribution == 2`, it will draw all results in the distribution
        way, and points from different time steps will be drawn with different colors.

        If `draw_distribution` received a `3-bit-like` digit (a integer value that
        bigger than 100, like `101` and `102`), the results
        will be saved on the video clip. 
        """
        return self._get('draw_distribution', 0, argtype='dynamic')

    @property
    def step(self) -> int:
        """
        Frame interval for sampling training data.
        """
        return self._get('step', 1, argtype='dynamic')

    @property
    def test_mode(self) -> str:
        """
        Test settings, canbe `'one'` or `'all'` or `'mix'`.
        When set it to `one`, it will test the model on the `args.force_split` only;
        When set it to `all`, it will test on each of the test dataset in `args.split`;
        When set it to `mix`, it will test on all test dataset in `args.split` together.
        """
        if not self.draw_results in ['null', '0', '1']:
            self._set('test_mode', 'one')

        return self._get('test_mode', 'mix', argtype='dynamic')

    @property
    def lr(self) -> float:
        """
        Learning rate.
        """
        return self._get('lr', 0.001, argtype='static')

    @property
    def K(self) -> int:
        """
        Number of multiple generations when test.
        This arg only works for `Generative Models`.
        """
        return self._get('K', 20, argtype='dynamic')

    @property
    def K_train(self) -> int:
        """
        Number of multiple generations when training.
        This arg only works for `Generative Models`.
        """
        return self._get('K_train', 10, argtype='static')

    @property
    def use_extra_maps(self) -> int:
        """
        Controls if uses the calculated trajectory maps or the given trajectory maps. 
        The model will load maps from `./dataset_npz/.../agent1_maps/trajMap.png`
        if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` 
        if set this argument to `1`.
        """
        return self._get('use_extra_maps', 0, argtype='dynamic')

    @property
    def dim(self) -> int:
        """
        Dimension of the `trajectory`.
        For example, (x, y) -> `dim = 2`.
        """
        if self.anntype == 'coordinate':
            dim = 2

        elif self.anntype == 'boundingbox':
            dim = 4

        else:
            raise ValueError(self.anntype)

        return self._get('dim', dim, argtype='static')

    @property
    def anntype(self) -> str:
        """
        Type of annotations in the predicted trajectories.
        Canbe `'coordinate'` or `'boundingbox'`.
        """
        return self._get('anntype', 'coordinate', argtype='static')

    @property
    def interval(self) -> float:
        """
        Time interval of each sampled trajectory coordinate.
        """
        return self._get('interval', 0.4, argtype='static')

    @property
    def pmove(self) -> int:
        """
        Index of the reference point when moving trajectories.
        """
        return self._get('pmove', -1, argtype='static')

    @property
    def pscale(self) -> str:
        """
        Index of the reference point when scaling trajectories.
        """
        return self._get('pscale', 'autoref', argtype='static')

    @property
    def protate(self) -> float:
        """
        Reference degree when rotating trajectories.
        """
        return self._get('protate', 0.0, argtype='static')

    @property
    def update_saved_args(self) -> int:
        """
        Choose if update (overwrite) json arg files or not.
        """
        return self._get('update_saved_args', 0, argtype='dynamic')
    