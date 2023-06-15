"""
@Author: Conghao Wong
@Date: 2023-05-30 09:26:04
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-15 15:49:16
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels import layers, transformer
from codes.managers import Structure
from codes.training.loss import ADE_2D

from ..__layers import OuterLayer
from ..agents import AgentArgs, BaseAgentModel, BaseAgentStructure


class Sieger300GhostModel(BaseAgentModel):

    def __init__(self, Args: AgentArgs,
                 as_single_model: bool = True,
                 structure: Structure = None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Layers
        # Linear prediction
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
        self.te0 = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                       transform_layer=self.t0)

        # Bilinear structure (outer product + pooling + fc)
        self.outer0 = OuterLayer(self.d//2, self.d//2, reshape=False)
        self.pooling0 = layers.MaxPooling2D(pool_size=(2, 2),
                                            data_format='channels_first')
        self.flatten0 = layers.Flatten(axes_num=2)
        self.outer_fc0 = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Random id encoding
        self.ie0 = layers.TrajEncoding(self.d//2, tf.nn.tanh)
        self.concat0 = tf.keras.layers.Concatenate(axis=-1)

        # Transformer is used as a feature extractor
        self.T0 = transformer.Transformer(num_layers=4,
                                          d_model=self.d,
                                          num_heads=8,
                                          dff=512,
                                          input_vocab_size=None,
                                          target_vocab_size=None,
                                          pe_input=self.Tsteps,
                                          pe_target=self.Tsteps,
                                          include_top=False)

        # Trainable adj matrix and gcn layer
        # It is used to generate multi-style predictions
        self.adj_fc0 = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.gcn0 = layers.GraphConv(units=self.d)

        # Decoder layers (with spectrums)
        self.decoder_fc0 = tf.keras.layers.Dense(self.d, tf.nn.tanh)
        self.decoder_fc1 = tf.keras.layers.Dense(
            self.Tsteps_de * self.Tchannels_de)
        self.decoder_reshape0 = tf.keras.layers.Reshape([self.args.Kc,
                                                         self.Tsteps_de,
                                                         self.Tchannels_de])

    def call(self, inputs, training=None, *args, **kwargs):

        # Unpack inputs
        obs = inputs[0]

        kp_future, kp_past = self.call_cell(obs, training, *args, **kwargs)

        return kp_future, obs, kp_past

    def call_cell(self, obs: tf.Tensor, training=None, *args, **kwargs):
        # Linear predict
        pred_linear = self.linear0(obs)     # (batch, pred, dim)
        traj = self.traj_concat0([obs, pred_linear])

        # Feature embedding and encoding
        # Use the bilinear structure to encode features
        f = self.te0(traj)          # (batch, Tsteps, d/2)
        f = self.outer0(f, f)       # (batch, Tsteps, d/2, d/2)
        f = self.pooling0(f)        # (batch, Tsteps, d/4, d/4)
        f = self.flatten0(f)        # (batch, Tsteps, d*d/16)
        f_bi = self.outer_fc0(f)            # (batch, Tsteps, d/2)

        # Sample random predictions
        p_all = []
        repeats = self.args.K_train if training else self.args.K

        for _ in range(repeats):
            # Assign random noise and embedding
            z = tf.random.normal(
                list(tf.shape(obs)[:-2]) + [self.Tsteps, self.d_id])
            f_z = self.ie0(z)

            # Feed into transformer
            f_tran = self.concat0([f_bi, f_z])

            f_behavior, _ = self.T0(inputs=f_tran,
                                    targets=self.t0(traj),
                                    training=training)

            # Multiple generations
            adj = self.adj_fc0(f_tran)
            i = list(range(adj.ndim))
            adj = tf.transpose(adj, i[:-2] + [i[-1], i[-2]])
            f_multi = self.gcn0(f_behavior, adj)

            # Forecast keypoints
            f_p = self.decoder_fc0(f_multi)
            f_p = self.decoder_fc1(f_p)
            f_p = tf.reshape(f_p, list(tf.shape(f_p)[:-1]) +
                             [self.Tsteps_de, self.Tchannels_de])

            # Inverse transform
            y = self.it0(f_p)
            p_all.append(y)

        Y = tf.concat(p_all, axis=-3)
        return Y[..., self.n_key_past:, :], Y[..., :self.n_key_past, :]


class Sieger300Ghost(BaseAgentStructure):

    def __init__(self, terminal_args: list[str], manager: Structure = None):
        super().__init__(terminal_args, manager)

        self.set_model_type(Sieger300GhostModel)

        if self.args.loss == 'keyl2':
            self.loss.set({self.keyl2: 1.0, self.keyl2_past: 1.0})
        elif self.args.loss == 'avgkey':
            self.loss.set({self.avgKey: 1.0})
        else:
            raise ValueError(self.args.loss)

    def keyl2_past(self, outputs: list[tf.Tensor],
                   labels: list[tf.Tensor],
                   coe: float = 1.0,
                   *args, **kwargs):

        if self.model.n_key_past:
            labels_pickled_past = tf.gather(
                outputs[1],
                self.model.key_indices_past,
                axis=-2)
            ade_past = ADE_2D(outputs[2], labels_pickled_past, coe=coe)
            return ade_past
        else:
            return 0
