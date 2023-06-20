"""
@Author: Conghao Wong
@Date: 2023-06-20 16:48:45
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-20 20:46:17
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels import layers, transformer
from codes.managers import Structure
from codes.utils import get_mask

from .. import loss
from ..__layers import OuterLayer
from ..agents import AgentArgs, BaseAgentModel, BaseAgentStructure


class AlphaModel(BaseAgentModel):

    def __init__(self, Args: AgentArgs, as_single_model: bool = True,
                 structure: Structure = None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        self.set_preprocess()

        if self.args.model_type != 'frame-based':
            self.log('This model only support model type `"frame-based"`',
                     level='error', raiseError=ValueError)

        # Layers
        self.linear0 = layers.LinearLayerND(obs_frames=self.args.obs_frames,
                                            pred_frames=self.args.pred_frames,
                                            diff=0.95)
        self.traj_concat0 = tf.keras.layers.Concatenate(axis=-2)

        # Transform
        Tlayer, ITlayer = layers.get_transform_layers(self.args.T)
        self.t0 = Tlayer(
            (self.args.obs_frames + self.args.pred_frames, self.dim))
        self.it0 = ITlayer((self.n_key, self.dim))
        self.Tsteps, self.Tchannels = self.t0.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it0.Tshape

        # Trajectory encoding
        self.flatten0 = layers.Flatten(axes_num=2)
        self.te0 = layers.TrajEncoding(self.d, tf.nn.relu)

        # Bilinear structure (outer product + pooling + fc)
        self.outer0 = OuterLayer(self.d, self.d, reshape=False)
        self.pooling0 = layers.MaxPooling2D(pool_size=(4, 4),
                                            data_format='channels_first')
        self.flatten1 = layers.Flatten(axes_num=2)
        self.outer_fc0 = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Random id encoding
        self.concat0 = tf.keras.layers.Concatenate(axis=-1)
        self.ie0 = layers.TrajEncoding(self.d//2, tf.nn.tanh)

        # Transformer is used as a feature extractor
        self.T0 = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=None,
            target_vocab_size=None,
            pe_input=self.Tsteps*self.Tchannels,
            pe_target=self.Tsteps*self.Tchannels,
            include_top=False)

        # Trainable adj matrix and gcn layer
        # It is used to generate multi-style predictions
        self.adj_fc0 = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.gcn0 = layers.GraphConv(units=self.d)

        # Decoder layers (with spectrums)
        self.decoder_fc0 = tf.keras.layers.Dense(self.d, tf.nn.tanh)
        self.decoder_fc1 = tf.keras.layers.Dense(
            self.args.max_agents * self.Tsteps_de * self.Tchannels_de)
        self.padding0 = layers.Padding(axis=-4)

    def call(self, inputs, training=None, *args, **kwargs):

        # Unpack inputs
        obs = inputs[0]         # (b:=batch, a:=agents, obs, dim)

        # mask shape: (b, a, 1, 1)
        mask = get_mask(tf.reduce_sum(obs, axis=[-1, -2]))
        mask = mask[..., tf.newaxis, tf.newaxis]
        obs = obs * mask

        # Linear predict
        pred_linear = self.linear0(obs)                 # (b, a, pred, dim)
        traj = self.traj_concat0([obs, pred_linear])    # (b, a, obs+pred, dim)

        # Reshape trajectories
        traj_s = self.t0(traj)                   # (b, a, steps, channels)
        traj_s = self.flatten0(traj_s)           # (b, a, steps*channels)
        traj_s = tf.transpose(traj_s, [0, 2, 1])  # (b, sc:=steps*channels, a)

        # Feature embedding and encoding
        # Use the bilinear structure to encode features
        f = self.te0(traj_s)        # (b, sc, d)
        f = self.outer0(f, f)       # (b, sc, d, d)
        f = self.pooling0(f)        # (b, sc, d/2, d/2)
        f = self.flatten1(f)        # (b, sc, d*d/16)
        f_bi = self.outer_fc0(f)    # (b, sc, d/2)

        # Sampling random noise vectors
        p_all = []
        repeats = self.args.K_train if training else self.args.K
        for _ in range(repeats):
            z = tf.random.normal(list(tf.shape(f_bi)[:-1]) + [self.d_id])
            f_z = self.ie0(z)

            f_behavior = self.concat0([f_bi, f_z])  # (b, sc, d)
            f_tran, _ = self.T0(inputs=f_behavior,
                                targets=traj_s,
                                training=training)

            # Multiple generations
            adj = self.adj_fc0(f_behavior)
            i = list(range(adj.ndim))
            adj = tf.transpose(adj, i[:-2] + [i[-1], i[-2]])
            f_multi = self.gcn0(f_tran, adj)    # (b, Kc, d)

            # Forecast keypoints
            f_p = self.decoder_fc0(f_multi)     # (b, Kc, d)
            f_p = self.decoder_fc1(f_p)         # (b, Kc, a*sc)
            f_p = tf.reshape(
                f_p, list(tf.shape(f_p)[:-1]) +
                [self.args.max_agents,
                 self.Tsteps_de,
                 self.Tchannels_de])            # (b, Kc, a, s, c)

            # Inverse transform
            y = self.it0(f_p)           # (b, Kc, a, n_key, dim)
            y = tf.transpose(y, [0, 2, 1, 3, 4])
            p_all.append(y)

        Y = tf.concat(p_all, axis=-3)   # (b, a, Kc, n_key, dim)
        return [Y[..., self.n_key_past:, :],
                Y[..., :self.n_key_past, :]]


class AlphaStructure(BaseAgentStructure):

    def __init__(self, terminal_args: list[str], manager: Structure = None, as_single_model: bool = True):
        super().__init__(terminal_args, manager, as_single_model)
        self.set_model_type(AlphaModel)
        self.loss.set({loss.keyl2: 1.0, loss.keyl2_past: 1.0})
