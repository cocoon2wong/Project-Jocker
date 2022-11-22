"""
@Author: Conghao Wong
@Date: 2022-06-20 21:40:38
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-22 09:27:27
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import layers, transformer

from ..__args import AgentArgs
from ..__layers import OuterLayer, get_transform_layers
from .__baseAgent import BaseAgentModel, BaseAgentStructure


class Agent47CModel(BaseAgentModel):

    def __init__(self, Args: AgentArgs,
                 feature_dim: int = 128,
                 id_depth: int = 16,
                 keypoints_number: int = 3,
                 keypoints_index: tf.Tensor = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, feature_dim, id_depth,
                         keypoints_number, keypoints_index,
                         structure, *args, **kwargs)

        # Layers
        self.Tlayer, self.ITlayer = get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = self.Tlayer(Oshape=(self.args.obs_frames, self.args.dim))
        self.it1 = self.ITlayer(Oshape=(self.n_key, self.args.dim))

        # Trajectory encoding
        self.te = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                      transform_layer=self.t1)

        # steps and shapes after applying transforms
        self.Tsteps_en = self.te.Tlayer.Tshape[0] if self.te.Tlayer else self.args.obs_frames
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Bilinear structure (outer product + pooling + fc)
        self.outer = OuterLayer(self.d//2, self.d//2, reshape=False)
        self.pooling = layers.MaxPooling2D(pool_size=(2, 2),
                                           data_format='channels_first')
        self.outer_fc = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Random id encoding
        self.ie = layers.TrajEncoding(self.d//2, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Transformer is used as a feature extractor
        self.T = transformer.Transformer(num_layers=4,
                                         d_model=self.d,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=None,
                                         pe_input=self.Tsteps_en,
                                         pe_target=self.Tsteps_en,
                                         include_top=False)

        # Trainable adj matrix and gcn layer
        # It is used to generate multi-style predictions
        self.adj_fc = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.gcn = layers.GraphConv(units=self.d)

        # Decoder layers (with spectrums)
        self.decoder_fc1 = tf.keras.layers.Dense(self.d, tf.nn.tanh)
        self.decoder_fc2 = tf.keras.layers.Dense(
            self.Tsteps_de * self.Tchannels_de)
        self.decoder_reshape = tf.keras.layers.Reshape(
            [self.args.Kc, self.Tsteps_de, self.Tchannels_de])

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None):
        """
        Run the first stage `agent47C` model.

        :param inputs: a list of tensors, including `trajs`
            - a batch of observed trajs, shape is `(batch, obs, 2)`

        :param training: set to `True` when training, or leave it `None`

        :return predictions: predicted keypoints, \
            shape = `(batch, Kc, N_key, 2)`
        """

        # unpack inputs
        trajs = inputs[0]   # (batch, obs, 2)
        bs = trajs.shape[0]

        # feature embedding and encoding -> (batch, Tsteps, d/2)
        # uses bilinear structure to encode features
        f = self.te.call(trajs)             # (batch, Tsteps, d/2)
        f = self.outer.call(f, f)           # (batch, Tsteps, d/2, d/2)
        f = self.pooling(f)                 # (batch, Tsteps, d/4, d/4)
        f = tf.reshape(f, [f.shape[0], f.shape[1], -1])
        spec_features = self.outer_fc(f)    # (batch, Tsteps, d/2)

        # Sample random predictions
        all_predictions = []
        rep_time = self.args.K_train if training else self.args.K

        t_outputs = self.t1.call(trajs)  # (batch, Tsteps, Tchannels)

        for _ in range(rep_time):
            # assign random ids and embedding -> (batch, Tsteps, d)
            ids = tf.random.normal([bs, self.Tsteps_en, self.d_id])
            id_features = self.ie.call(ids)

            # transformer inputs
            # shapes are (batch, Tsteps, d)
            t_inputs = self.concat([spec_features, id_features])

            # transformer -> (batch, Tsteps, d)
            behavior_features, _ = self.T.call(inputs=t_inputs,
                                               targets=t_outputs,
                                               training=training)

            # multi-style features -> (batch, Kc, d)
            adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
            m_features = self.gcn.call(behavior_features, adj)

            # predicted keypoints -> (batch, Kc, Tsteps_Key, Tchannels)
            y = self.decoder_fc1(m_features)
            y = self.decoder_fc2(y)
            y = self.decoder_reshape(y)

            y = self.it1.call(y)
            all_predictions.append(y)

        return tf.concat(all_predictions, axis=1)


class Agent47C(BaseAgentStructure):

    """
    Training structure for the `Agent47C` model.
    Note that it is only used to train the single model.
    Please use the `Silverballers` structure if you want to test any
    agent-handler based silverballers models.
    """

    def __init__(self, terminal_args: list[str], manager=None):
        super().__init__(terminal_args, manager)

        self.set_model_type(new_type=Agent47CModel)
