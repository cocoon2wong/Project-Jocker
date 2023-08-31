"""
@Author: Conghao Wong
@Date: 2022-09-01 10:38:49
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-31 09:32:55
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...base import BaseObject
from ...constant import OUTPUT_TYPES
from ...dataset import Annotation


class BaseProcessLayer(tf.keras.layers.Layer, BaseObject):

    def __init__(self, anntype: str,
                 preprocess_input_types: list[str],
                 postprocess_input_types: list[str]):

        tf.keras.layers.Layer.__init__(self)
        BaseObject.__init__(self, name=self.name)

        self.anntype = anntype
        self.preprocess_input_types = preprocess_input_types
        self.postprocess_input_types = postprocess_input_types

        self.picker = Annotation(anntype) if anntype else None

    def call(self, inputs: dict[str, tf.Tensor],
             preprocess: bool,
             update_paras=False,
             training=None, *args, **kwargs) -> dict[str, tf.Tensor]:
        """
        Run preprocess or postprocess on the input dictionary.
        """
        if preprocess:
            if update_paras:
                self.update_paras(inputs)
            outputs = self.preprocess(inputs)
        else:
            outputs = self.postprocess(inputs)

        return outputs

    def preprocess(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        raise NotImplementedError('Please rewrite this method')

    def postprocess(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        raise NotImplementedError('Please rewrite this method')

    def update_paras(self, inputs: dict[str, tf.Tensor]) -> None:
        raise NotImplementedError('Please rewrite this method')


class ProcessModel(tf.keras.Model):

    def __init__(self, layers: list[BaseProcessLayer], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.players = layers
        self.preprocess_input_types: list[str] = None
        self.postprocess_input_types = [OUTPUT_TYPES.PREDICTED_TRAJ]

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
