"""
@Author: Conghao Wong
@Date: 2023-07-10 15:21:12
@LastEditors: Beihao Xia
@LastEditTime: 2023-07-11 16:55:38
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf

from codes.basemodels import layers, transformer
from codes.constant import INPUT_TYPES
from codes.utils import get_mask

from ..__layers import OuterLayer
from ..agents import AgentArgs, BaseAgentModel, BaseAgentStructure


class BetaModel(BaseAgentModel):

    def __init__(self, Args: AgentArgs, as_single_model: bool = True,
                 structure: BaseAgentStructure = None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Assign model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        tlayer, itlayer = layers.get_transform_layers(self.args.T)

        # Transform layers
        self.ts = tlayer((self.args.obs_frames, 2))
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((self.n_key, self.dim))

        # Trajectory encoding
        self.te = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                      transform_layer=self.t1)

        # SocialCircle encoding
        self.tse = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                       transform_layer=self.ts)

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Bilinear structure (outer product + pooling + fc)
        # For trajectories
        self.outer = OuterLayer(self.d//2, self.d//2)
        self.pooling = layers.MaxPooling2D(
            (2, 2), data_format='channels_first')
        self.flatten = layers.Flatten(axes_num=2)
        self.outer_fc = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d//2, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.concat_fc = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Transformer backbone
        self.T = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=None,
            target_vocab_size=None,
            pe_input=self.Tsteps_en,
            pe_target=self.Tsteps_en,
            include_top=False
        )

        # Multi-style prediction
        self.ms_fc = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.ms_conv = layers.GraphConv(self.d)

        # Decoder layers
        self.decoder_fc1 = tf.keras.layers.Dense(self.d, tf.nn.tanh)
        self.decoder_fc2 = tf.keras.layers.Dense(
            self.Tsteps_de * self.Tchannels_de
        )

    def call(self, inputs, training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        obs = inputs[0]     # (batch, obs, dim)
        nei = inputs[1]     # (batch, a:=max_agents, obs, dim)

        # Start computing the SocialCircle
        # SocialCircle will be computed on each agent's center point
        c_obs = self.picker.get_center(obs)[..., :2]
        c_nei = self.picker.get_center(nei)[..., :2]

        # Move vectors
        nei_vector = c_nei[..., -1, :] - c_nei[..., 0, :]       # (batch, a, 2)
        obs_vector = c_obs[..., -1:, :] - c_obs[..., 0:1, :]    # (batch, 1, 2)

        # Calculate neighbors' relative move angles
        nei_angle = tf.atan2(x=nei_vector[..., 0], y=nei_vector[..., 1])
        obs_angle = tf.atan2(x=obs_vector[..., 0], y=obs_vector[..., 1])
        rel_angle = nei_angle - obs_angle

        # Calculate relative walking distances with all neighbors
        nei_vector_len = tf.linalg.norm(nei_vector, axis=-1)    # (batch, a)
        obs_vector_len = tf.linalg.norm(obs_vector, axis=-1)    # (batch, 1)
        rel_vector_len = (nei_vector_len + 0.0001) / (obs_vector_len + 0.0001)

        # Calculate distances between neighbors and the target agent
        nei_posion_vector = c_nei[..., -1, :] - c_obs[..., tf.newaxis, -1, :]
        nei_distance = tf.linalg.norm(nei_posion_vector, axis=-1)
        nei_posion_angle = tf.atan2(x=nei_posion_vector[..., 0],
                                    y=nei_posion_vector[..., 1])
        nei_posion_angle = tf.math.mod(nei_posion_angle, 2*np.pi)
        angle_indices = nei_posion_angle / (2*np.pi/self.args.obs_frames)
        angle_indices = tf.cast(angle_indices, tf.int32)

        # Mask neighbors
        nei_mask = get_mask(tf.reduce_sum(nei, axis=[-1, -2]), tf.int32)
        angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

        # Compute SocialCircle
        social_circle = []
        for ang in range(self.args.obs_frames):
            _mask = tf.cast(angle_indices == ang, tf.float32)
            _mask_count = tf.reduce_sum(_mask, axis=-1)

            n = _mask_count + 0.0001
            avg_len = tf.reduce_sum(rel_vector_len * _mask, axis=-1) / n
            avg_disance = tf.reduce_sum(nei_distance * _mask, axis=-1) / n
            avg_angle = tf.reduce_sum(rel_angle * _mask, axis=-1) / n
            social_circle.append([avg_len, avg_disance, avg_angle])

        # Shape of the final SocialCircle: (batch, obs, 2)
        social_circle = tf.cast(social_circle, tf.float32)
        social_circle = tf.transpose(social_circle, [2, 0, 1])

        # Encode the SocialCircle
        f_social = self.tse(social_circle)    # (batch, steps, d/2)

        # Trajectory embedding and encoding
        f = self.te(obs)
        f = self.outer(f, f)
        f = self.pooling(f)
        f = self.flatten(f)
        f_traj = self.outer_fc(f)       # (batch, steps, d/2)

        # Feature fusion
        f_behavior = tf.concat([f_traj, f_social], axis=-1)
        f_behavior = self.concat_fc(f_behavior)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K
        traj_targets = self.t1(obs)

        for _ in range(repeats):
            z = tf.random.normal(list(tf.shape(f_behavior)[:-1]) + [self.d_id])
            f_z = self.ie(z)
            # (batch, steps, 2*d)
            f_final = tf.concat([f_behavior, f_z], axis=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Multiple generations
            adj = self.ms_fc(f_final)      # (batch, steps, Kc)
            i = list(tf.range(adj.ndim))
            adj = tf.transpose(adj, i[:-2] + [i[-1], i[-2]])
            f_multi = self.ms_conv(f_tran, adj)    # (b, Kc, d)

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)
            y = tf.reshape(y, list(tf.shape(y)[:-1]) +
                           [self.Tsteps_de, self.Tchannels_de])

            y = self.it1(y)
            all_predictions.append(y)

        Y = tf.concat(all_predictions, axis=-3)   # (batch, K, n_key, dim)
        return Y


class BetaStructure(BaseAgentStructure):
    def __init__(self, terminal_args: list[str], manager=None, as_single_model: bool = True):
        terminal_args += ['--model_type', 'agent-based']
        super().__init__(terminal_args, manager, as_single_model)
        self.set_model_type(BetaModel)
