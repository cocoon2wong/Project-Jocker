"""
@Author: Conghao Wong
@Date: 2022-06-20 16:14:03
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-06 15:50:51
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import time
from typing import TypeVar

import numpy as np
import tensorflow as tf

from ..args import Args
from ..base import BaseManager
from ..constant import INPUT_TYPES
from ..utils import CHECKPOINT_FILENAME, WEIGHTS_FORMAT
from . import process

T = TypeVar('T')

MAX_INFERENCE_TIME_STORGED = 100


class Model(tf.keras.Model, BaseManager):
    """
    Model (Model Manager)
    -----

    Usage
    -----
    When training or testing new models, please subclass this class, and clarify
    model layers used in your model.
    ```python
    class MyModel(Model):
        def __init__(self, Args, structure, *args, **kwargs):
            super().__init__(Args, structure, *args, **kwargs)

            self.fc = tf.keras.layers.Dense(64, tf.nn.relu)
            self.fc1 = tf.keras.layers.Dense(2)
    ```

    Then define your model's pipeline in `call` method:
    ```python
        def call(self, inputs, training=None, mask=None):
            y = self.fc(inputs)
            return self.fc1(y)
    ```

    Public Methods
    --------------
    ```python
    # forward model with pre-process and post-process
    (method) forward: (self: Self@Model,
                       inputs: list[Tensor],
                       training: Any | None = None) -> list[Tensor]

    # Set model inputs
    (method) set_inputs: (self: Self@Model, *args: Any) -> None

    # Set pre/post-process methods
    (method) set_preprocess: (self: Self@Model, **kwargs: Any) -> None
    ```
    """

    def __init__(self, Args: Args,
                 structure=None,
                 *args, **kwargs):

        tf.keras.Model.__init__(self, *args, **kwargs)
        BaseManager.__init__(self, manager=structure, name=self.name)

        # Pre/post-process model and settings
        self.processor = process.ProcessModel(self.args)

        # Model inputs
        self.input_types: list[str] = []
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ)

        # Inference times
        self.inference_times: list[float] = []

        # Extra model outputs and their indexes
        self.ext_traj_wise_outputs: dict[int, str] = {}
        self.ext_agent_wise_outputs: dict[int, str] = {}

    @property
    def structure(self) -> BaseManager:
        return self.manager

    @structure.setter
    def structure(self, value: T) -> T:
        self.manager = value

    @property
    def average_inference_time(self) -> int:
        """
        Average inference time (ms).
        """
        if l := len(it := self.inference_times):
            if l > 3:
                it = it[1:-1]
            t = np.mean(it)
            return int(1000 * t)
        else:
            return '(Not Available)'

    @property
    def fastest_inference_time(self) -> int:
        """
        The fastest inference time (ms).
        """
        if l := len(it := self.inference_times):
            if l > 3:
                it = it[1:-1]
            t = min(it)
            return int(1000 * t)
        else:
            return '(Not Available)'

    def call(self, inputs,
             training=None,
             mask=None,
             *args, **kwargs):

        raise NotImplementedError

    def forward(self, inputs: list[tf.Tensor],
                training=None) -> list[tf.Tensor]:
        """
        Run a forward implementation.

        :param inputs: Input tensor (or a `list` of tensors).
        :param training: Config if running as training or test mode.
        :return outputs_p: Model's output. type=`list[tf.Tensor]`.
        """
        # Preprocess
        inputs_p = self.processor(inputs, preprocess=True, training=training)

        # Model inference
        time_start = time.time()
        outputs = self(inputs_p, training=training)
        time_end = time.time()

        # Compute time costs
        l = MAX_INFERENCE_TIME_STORGED
        if len(self.inference_times) > l:
            self.inference_times = self.inference_times[l//2:]

        time_cost = time_end - time_start
        self.inference_times.append(time_cost)

        # Postprocess
        return self.processor(outputs, preprocess=False, training=training)

    def set_inputs(self, *args):
        """
        Set input types of the model.
        Accept keywords:
        ```python
        codes.constant.INPUT_TYPES.OBSERVED_TRAJ
        codes.constant.INPUT_TYPES.MAP
        codes.constant.INPUT_TYPES.DESTINATION_TRAJ
        codes.constant.INPUT_TYPES.GROUNDTRUTH_TRAJ
        codes.constant.INPUT_TYPES.GROUNDTRUTH_SPECTRUM
        codes.constant.INPUT_TYPES.ALL_SPECTRUM
        ```

        :param input_names: Type = `str`, accept several keywords.
        """
        self.input_types = [item for item in args]
        self.processor.set_preprocess_input_types(self.input_types)

    def set_preprocess(self, *args, **kwargs):
        """
        Set pre-process methods used before training.
        For example, enable all three preprocess methods by
        ```python
        self.set_preprocess(PROCESS_TYPES.MOVE,
                            PROCESS_TYPES.ROTATE,
                            PROCESS_TYPES.SCALE)
        ```

        *IMPORTANT NOTE:* You MUST call `self.set_inputs` after the
        `set_preprocess` to initialize all process layers.

        Extra args:
        :param builtin: Controls whether preprocess methods applied
            outside from the `call` method. If `builtin == True`, the
            preprocess layer will be called outside from the `call`.
        """
        return self.processor.set_process(*args, **kwargs)

    def load_weights_from_logDir(self, weights_dir: str):
        all_files = os.listdir(weights_dir)
        weights_files = [f for f in all_files
                         if WEIGHTS_FORMAT + '.' in f]
        weights_files.sort()

        if CHECKPOINT_FILENAME in all_files:
            p = os.path.join(weights_dir, CHECKPOINT_FILENAME)
            epoch = int(np.loadtxt(p)[1])

            weights_files = [f for f in weights_files
                             if f'_epoch{epoch}{WEIGHTS_FORMAT}' in f]

        weights_name = weights_files[-1].split('.index')[0]
        self.load_weights(p := os.path.join(weights_dir, weights_name))
        self.log(f'Successfully load weights from `{p}`.', verbose_mode=True)

    def print_info(self, **kwargs):
        try:
            p_layers = [l.name for l in self.processor.layers]
        except:
            p_layers = None

        info = {'Model type': type(self).__name__,
                'Model name': self.args.model_name,
                'Model prediction type': self.args.anntype,
                'Preprocess used': p_layers}

        kwargs.update(**info)
        return super().print_info(**kwargs)
