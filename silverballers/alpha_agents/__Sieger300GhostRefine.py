"""
@Author: Conghao Wong
@Date: 2023-06-08 15:59:32
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-13 19:02:43
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import math

import tensorflow as tf

from codes.managers import SecondaryBar, Structure

from ..agents import AgentArgs
from .__Sieger300Ghost import Sieger300GhostModel


class Sieger300GhostRefineModel(Sieger300GhostModel):

    def __init__(self, Args: AgentArgs,
                 as_single_model: bool = True,
                 structure: Structure = None,
                 *args, **kwargs):
        
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        if self.args.model_type != 'frame-based':
            self.log('This model only supports type `fame-based`.'+
                     f'Current setting is `{self.args.model_type}`.',
                     level='error', raiseError=ValueError)

    def call(self, inputs, training=None, *args, **kwargs):

        # Unpack inputs
        obs = inputs[0]

        current_batch_size = self.args.max_agents
        expected_batch_size = self.args.batch_size
        threads = math.ceil(expected_batch_size / current_batch_size)
        all_batch = len(obs)
        run_count = math.ceil(all_batch/threads)
        
        # Run cells
        kp_past = []
        kp_future = []

        obs = list(obs)
        for _i in SecondaryBar(range(run_count), manager=self.get_top_manager(), desc='Calculate on batch...'):
            
            start = _i * threads
            end = min(all_batch, (_i + 1) * threads)
            _obs = tf.concat(obs[start:end], axis=-3)

            _kp_future, _kp_past = self.call_cell(_obs, training, *args, **kwargs)
      
            kp_past.append(_kp_past)
            kp_future.append(_kp_future)

        kp_future = tf.concat(kp_future, axis=0)
        kp_past = tf.concat(kp_past, axis=0)

        kp_future = tf.reshape(kp_future, [all_batch, current_batch_size] + kp_future.shape[1:])
        kp_past = tf.reshape(kp_past, [all_batch, current_batch_size] + kp_past.shape[1:])

        return kp_future, obs, kp_past

    