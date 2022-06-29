"""
@Author: Conghao Wong
@Date: 2022-06-22 20:00:17
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-29 10:45:26
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import Model, layers, transformer
from codes.training import loss

from ..__args import AgentArgs
from ..__layers import OuterLayer, get_transform_layers
from .__baseAgent import BaseAgentStructure


class Agent47CEModel(Model):

    def __init__(self, Args: AgentArgs,
                 feature_dim: int = 128,
                 id_depth: int = 16,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.args = Args

        # Parameters
        self.d = feature_dim
        self.d_id = id_depth

        # Preprocess
        preprocess_list = ()
        for index, operation in enumerate(['Move', 'Scale', 'Rotate']):
            if self.args.preprocess[index] == '1':
                preprocess_list += (operation,)

        self.set_preprocess(*preprocess_list)
        self.set_preprocess_parameters(move=0)

        # Layers
        self.Tlayer, self.ITlayer = get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = self.Tlayer((self.args.obs_frames, self.args.dim))
        self.it1 = self.ITlayer((self.args.pred_frames, self.args.dim))

        # Trajectory encoding (with FFTs)
        self.te = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                      transform_layer=self.t1)

        # steps and shapes after applying transforms
        self.Tsteps_en = self.t1.Tshape[0]
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Bilinear structure (outer product + pooling + fc)
        self.outer = OuterLayer(self.d//2, self.d//2, reshape=False)
        self.pooling = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
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
        self.gcn = layers.GraphConv(units=2*self.d)

        # Decoder layers (with spectrums)
        self.decoder_fc1 = tf.keras.layers.Dense(2*self.d, tf.nn.tanh)
        self.decoder_fc2 = tf.keras.layers.Dense(
            self.Tsteps_de * self.Tchannels_de)
        self.decoder_reshape = tf.keras.layers.Reshape(
            [self.args.Kc, self.Tsteps_de, self.Tchannels_de])

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None):
        """
        Run the first stage `agent47CE` model.

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

            # predicted keypoints -> (batch, Kc, pred, 2)
            y = self.decoder_fc1(m_features)
            y = self.decoder_fc2(y)
            y = self.decoder_reshape(y)

            y = self.it1.call(y)
            all_predictions.append(y)

        return tf.concat(all_predictions, axis=1)


class Agent47CE(BaseAgentStructure):

    """
    Training structure for the `Agent47CE` model.
    Note that it is only used to train the single model.
    Please use the `Silverballers` structure if you want to test any
    agent-handler based silverballers models.
    """

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.set_model_type(new_type=Agent47CEModel)
        self.important_args += ['T']

        self.set_loss(self.avg_keypoint_loss, self.avg_pointwise_loss)
        self.set_loss_weights(0.7, 0.3)

    def avg_keypoint_loss(self, outputs: list[tf.Tensor],
                          labels: tf.Tensor,
                          *args, **kwargs) -> tf.Tensor:
        """
        Average keypoint loss.
        The average l2 distance between keypoints and predictions.
        """

        labels_pickled = tf.gather(labels, self.p_index, axis=1)
        pred_pickled = tf.gather(outputs[0], self.p_index, axis=-2)

        return loss.ADE(pred=pred_pickled, GT=labels_pickled)

    def avg_pointwise_loss(self, outputs: list[tf.Tensor],
                           labels: tf.Tensor,
                           *args, **kwargs):
        """
        Average point-wise l2 loss.
        """

        return loss.ADE(pred=outputs[0], GT=labels)
