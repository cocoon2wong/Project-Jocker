"""
@Author: Conghao Wong
@Date: 2023-09-06 15:28:21
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-06 15:48:15
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...args import Args
from ...constant import OUTPUT_TYPES, PROCESS_TYPES
from .__base import BaseProcessLayer
from .__move import Move
from .__rotate import Rotate
from .__scale import Scale

PROCESS_DICT: dict[str, type[BaseProcessLayer]] = {
    PROCESS_TYPES.MOVE: Move,
    PROCESS_TYPES.SCALE: Scale,
    PROCESS_TYPES.ROTATE: Rotate}


class ProcessModel(tf.keras.Model):

    def __init__(self, Args: Args,
                 layers: list[BaseProcessLayer] = [],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.args = Args
        self.players = layers

        self.preprocess_input_types: list[str] = None
        self.postprocess_input_types = [OUTPUT_TYPES.PREDICTED_TRAJ]

        self.process_paras = {PROCESS_TYPES.MOVE: Args.pmove}

    def set_layers(self, layers: list[BaseProcessLayer] = [],
                   built_in: bool = True):
        """
        Set pre/post-process layers manually.

        :param layers: A list of pre/post-process layer objects.
        :param builtin: Choose whether to set layers inner the builtin \
            pre/post-process model or generate a new process model.
        """
        if built_in:
            self.players = layers
            return self
        else:
            return ProcessModel(self.args, layers)

    def set_process(self, *args, **kwargs):
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
        processors = []
        for p in [PROCESS_TYPES.MOVE,
                  PROCESS_TYPES.ROTATE,
                  PROCESS_TYPES.SCALE]:

            p_args = [self.args.anntype]
            p_kwargs = {}

            if p in args:
                if p in self.process_paras.keys():
                    p_args.append(self.process_paras[p])

            elif p in kwargs.keys():
                v = kwargs[p]
                if type(v) is dict:
                    p_kwargs = v
                else:
                    p_args.append(v)

            else:
                continue

            processors.append(PROCESS_DICT[p](*p_args, **p_kwargs))

        if 'builtin' in kwargs.keys():
            builtin = kwargs['builtin']
        else:
            builtin = True
        return self.set_layers(processors, builtin)

    def set_preprocess_input_types(self, types: list[str]):
        self.preprocess_input_types = types

    def set_postprocess_input_types(self, types: list[str]):
        self.postprocess_input_types = types

    def call(self, inputs: list[tf.Tensor],
             preprocess: bool,
             update_paras=True,
             training=None,
             *args, **kwargs) -> list[tf.Tensor]:

        if preprocess:
            layers = self.players
            type_var_name = 'preprocess_input_types'
            input_types = self.preprocess_input_types
        else:
            layers = self.players[::-1]
            type_var_name = 'postprocess_input_types'
            input_types = self.postprocess_input_types

        # Whether apply pre/post-process on other inputs and outputs
        if self.args.only_process_trajectory:
            input_types = input_types[:1]

        if type(inputs) not in [list, tuple]:
            inputs = [inputs]
        if type(inputs) is tuple:
            inputs = list(inputs)

        for p in layers:
            # Prepare tensors to be processed
            p_dict = {}
            for _type in getattr(p, type_var_name):
                if _type not in input_types:
                    continue

                value = inputs[input_types.index(_type)]
                p_dict[_type] = value

            # Run process layers
            p_outputs = p(p_dict, preprocess,
                          update_paras, training,
                          *args, **kwargs)

            # Put back processed tensors
            for _type, value in p_outputs.items():
                if _type in input_types:
                    index = input_types.index(_type)
                    inputs[index] = value

        return inputs
