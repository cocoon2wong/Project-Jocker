"""
@Author: Conghao Wong
@Date: 2023-05-30 09:26:04
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-30 11:13:10
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels import layers, transformer
from codes.managers import Structure

from ..__args import AgentArgs
from ..__layers import OuterLayer
from ..agents import BaseAgentModel, BaseAgentStructure


class Sieger300GhostModel(BaseAgentModel):

    def __init__(self, Args: AgentArgs,
                 feature_dim: int = 128,
                 id_depth: int = 16,
                 keypoints_number: int = 3,
                 keypoints_index: tf.Tensor = None,
                 structure=None, *args, **kwargs):

        super().__init__(Args, feature_dim, id_depth,
                         keypoints_number, keypoints_index,
                         structure, *args, **kwargs)

        # Layers
        # Linear prediction
        self.linear0 = layers.LinearLayerND(obs_frames=self.args.obs_frames,
                                            pred_frames=self.args.pred_frames,
                                            diff=0.95)

        # Transform
        Tlayer, ITlayer = layers.get_transform_layers(self.args.T)
        self.t0 = Tlayer((self.args.pred_frames, self.dim))
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

        # Linear predict
        pred_linear = self.linear0(obs)     # (batch, pred, dim)

        # Feature embedding and encoding
        # Use the bilinear structure to encode features
        f = self.te0(pred_linear)           # (batch, Tsteps, d/2)
        f = self.outer0(f, f)               # (batch, Tsteps, d/2, d/2)
        f = self.pooling0(f)                # (batch, Tsteps, d/4, d/4)

        # Flatten
        s = tf.shape(f)
        f = tf.reshape(f, [s[0], s[1], -1])  # (batch, Tsteps, d*d/16)
        f_bi = self.outer_fc0(f)            # (batch, Tsteps, d/2)

        # Sample random predictions
        p_all = []
        repeats = self.args.K_train if training else self.args.K

        for _ in range(repeats):
            # Assign random noise and embedding
            z = tf.random.normal([s[0], self.Tsteps, self.d_id])
            f_z = self.ie0(z)

            # Feed into transformer
            f_tran = self.concat0([f_bi, f_z])

            f_behavior, _ = self.T0(inputs=f_tran,
                                    targets=self.t0(pred_linear),
                                    training=training)

            # Multiple generations
            adj = tf.transpose(self.adj_fc0(f_tran), [0, 2, 1])
            f_multi = self.gcn0(f_behavior, adj)

            # Forecast keypoints
            f_p = self.decoder_fc0(f_multi)
            f_p = self.decoder_fc1(f_p)
            f_p = self.decoder_reshape0(f_p)

            # Inverse transform
            y = self.it0(f_p)
            p_all.append(y)

        return tf.concat(p_all, axis=1)


class Sieger300Ghost(BaseAgentStructure):

    def __init__(self, terminal_args: list[str], manager: Structure = None):
        super().__init__(terminal_args, manager)

        self.set_model_type(Sieger300GhostModel)
