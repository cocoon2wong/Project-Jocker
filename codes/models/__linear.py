"""
@Author: Conghao Wong
@Date: 2022-07-15 20:13:07
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-03 14:59:11
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from ..args import BaseArgTable
from ..basemodels import Model, layers
from ..training import Structure


class LinearModel(Model):
    def __init__(self, Args: BaseArgTable, structure=None, *args, **kwargs):
        super().__init__(Args, structure, *args, **kwargs)

        self.linear = layers.LinearLayerND(obs_frames=self.args.obs_frames,
                                           pred_frames=self.args.pred_frames)

    def call(self, inputs, training=None, *args, **kwargs):
        trajs = inputs[0]
        return self.linear.call(trajs)


class Linear(Structure):
    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.set_inputs('obs')
        self.set_labels('pred')

        self.args._set('epochs', 3)
        self.args._set('use_maps', 0)

        self.noTraining = True

    def create_model(self, *args, **kwargs) -> Model:
        return LinearModel(self.args)
