"""
@Author: Conghao Wong
@Date: 2023-06-20 16:48:45
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-26 10:35:23
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels import layers, process, transformer
from codes.managers import Structure
from codes.utils import get_mask

from .. import loss
from ..__layers import OuterLayer
from ..agents import AgentArgs, BaseAgentModel, BaseAgentStructure


class CenterMove(process.BaseProcessLayer):
    """
    Move the center of a frame of trajectories (at the reference moment)
    to $$\\vec{0}$$.
    The default reference time step is the last observation step.
    It is only used for `'frame-based'` models.
    """

    def __init__(self, anntype: str = None, ref: int = -1, *args, **kwargs):
        super().__init__(anntype, ref, *args, **kwargs)

        self.need_mask = True

    def update_paras(self, trajs: tf.Tensor) -> None:
        # (batch, agents, (K), steps, dim)
        frame = trajs[..., tf.newaxis, self.ref, :]

        # Compute mask, shape = (batch, agents, (K), 1, 1)
        mask = get_mask(frame)[..., 0:1]
        mask_count = tf.reduce_sum(mask, axis=1, keepdims=True)

        # Compute center (with mask)
        frame *= mask
        center = tf.reduce_sum(frame, axis=1, keepdims=True)
        center /= mask_count

        # Save parameters
        self.paras = center
        self.mask_paras = mask

    def preprocess(self, trajs: tf.Tensor, use_new_paras=True) -> tf.Tensor:
        if use_new_paras:
            self.update_paras(trajs)

        return self.move(trajs, center=self.paras)

    def postprocess(self, trajs: tf.Tensor) -> tf.Tensor:
        return self.move(trajs, center=self.paras, inverse=True)

    def move(self, trajs: tf.Tensor, center: tf.Tensor, inverse=False):
        while center.ndim < trajs.ndim:
            center = center[..., tf.newaxis, :, :]

        mask = self.mask_paras
        while mask.ndim < trajs.ndim:
            mask = mask[..., tf.newaxis, :, :]

        if inverse:
            center *= -1.0

        return (trajs - center) * mask


class AlphaModel(BaseAgentModel):

    def __init__(self, Args: AgentArgs, as_single_model: bool = True,
                 structure: Structure = None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        if not len(self.key_indices_past):
            self.structure.loss.set({loss.keyl2: 1.0})
        else:
            self.structure.loss.set({loss.keyl2: 1.0, loss.keyl2_past: 1.0})

        # Preprocess
        self.set_preprocess_layers([CenterMove()])

        preprocess = {}
        for index, operation in enumerate(['move', 'scale', 'rotate']):
            if self.args.preprocess[index] == '1':
                preprocess[operation] = 'auto'

        self.p_layer_ii = self.set_preprocess(builtin=False, **preprocess)

        # Layers
        self.linear0 = layers.LinearLayerND(obs_frames=self.args.obs_frames,
                                            pred_frames=self.args.pred_frames,
                                            diff=0.95)
        self.traj_concat0 = tf.keras.layers.Concatenate(axis=-2)

        # Transform
        Tlayer, ITlayer = layers.get_transform_layers(self.args.T)
        self.t0 = Tlayer((
            self.args.obs_frames + self.args.pred_frames,
            self.args.max_agents * self.dim))
        self.it0 = ITlayer((
            self.n_key,
            self.args.max_agents * self.dim))

        self.Tsteps, self.Tchannels = self.t0.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it0.Tshape

        # Trajectory encoding
        self.flatten0 = layers.Flatten(axes_num=2)
        self.te0 = layers.TrajEncoding(self.d, tf.nn.relu)
        self.te0_ii = layers.TrajEncoding(self.d, tf.nn.relu)

        # Bilinear structure (outer product + pooling + fc)
        self.outer0 = OuterLayer(self.d, self.d, reshape=False)
        self.pooling0 = layers.MaxPooling2D(pool_size=(4, 4),
                                            data_format='channels_first')
        self.flatten1 = layers.Flatten(axes_num=2)
        self.outer_fc0 = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)
        self.outer_fc0_ii = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

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
            self.Tsteps_de * self.Tchannels_de)
        self.padding0 = layers.Padding(axis=-4)

    def call(self, inputs, training=None, mask=None, *args, **kwargs):

        # Unpack inputs
        obs = inputs[0]         # (b:=batch, a:=agents, obs, dim)
        obs = obs * mask

        # Linear predict
        pred_linear = self.linear0(obs)                 # (b, a, pred, dim)
        traj = self.traj_concat0([obs, pred_linear])    # (b, a, obs+pred, dim)

        # Reshape and transform trajectories
        traj_r = tf.transpose(traj, [0, 2, 1, 3])   # (b, obs+pred, a, dim)
        traj_r = self.flatten0(traj_r)              # (b, obs+pred, a*dim)
        traj_s = self.t0(traj_r)                    # (b, steps, channels)

        # Feature embedding and encoding
        # Use the bilinear structure to encode features
        f = self.te0(traj_s)        # (b, steps, d)
        f = self.outer0(f, f)       # (b, steps, d, d)
        f = self.pooling0(f)        # (b, steps, d/4, d/4)
        f = self.flatten1(f)        # (b, steps, d*d/16)
        f_bi = self.outer_fc0(f)    # (b, steps, d/2)

        # Second branch
        obs_ii = self.p_layer_ii(inputs, preprocess=True,
                                 update_paras=True,
                                 training=training)[0]
        obs_ii *= mask
        pred_linear_ii = self.linear0(obs_ii)
        traj_ii = self.traj_concat0([obs_ii, pred_linear_ii])

        # (b, obs+pred, a, dim)
        traj_r_ii = tf.transpose(traj_ii, [0, 2, 1, 3])
        # (b, obs+pred, a*dim)
        traj_r_ii = self.flatten0(traj_r_ii)
        # (b, steps, channels)
        traj_s_ii = self.t0(traj_r_ii)

        f_ii = self.te0_ii(traj_s_ii)        # (b, steps, d)
        # f_ii = self.outer0(f_ii, f_ii)
        # f_ii = self.pooling0(f_ii)
        # f_ii = self.flatten1(f_ii)
        f_bi_ii = self.outer_fc0_ii(f_ii)    # (b, steps, d/2)

        # Sampling random noise vectors
        p_all = []
        repeats = self.args.K_train if training else self.args.K
        for _ in range(repeats):
            z = tf.random.normal(list(tf.shape(f_bi)[:-1]) + [self.d_id])
            f_z = self.ie0(z)

            f_behavior = self.concat0([f_bi, f_bi_ii, f_z])  # (b, steps, d)
            f_tran, _ = self.T0(inputs=f_behavior,
                                targets=traj_s,
                                training=training)

            # Multiple generations
            adj = self.adj_fc0(f_behavior)      # (b, steps, Kc)
            adj = tf.transpose(adj, [0, 2, 1])
            f_multi = self.gcn0(f_tran, adj)    # (b, Kc, d)

            # Forecast keypoints
            f_p = self.decoder_fc0(f_multi)     # (b, Kc, d)
            f_p = self.decoder_fc1(f_p)         # (b, Kc, steps*channels)
            f_p = tf.reshape(
                f_p, list(tf.shape(f_p)[:-1]) +
                [self.Tsteps_de,
                 self.Tchannels_de])            # (b, Kc, steps, channels)

            # Inverse transform
            y = self.it0(f_p)                   # (b, Kc, n_key, a*dim)
            y = tf.reshape(
                y, list(tf.shape(y))[:-1] +
                [self.args.max_agents, self.dim])   # (b, Kc, n_key, a, dim)
            y = tf.transpose(y, [0, 3, 1, 2, 4])
            p_all.append(y)

        Y = tf.concat(p_all, axis=-3)   # (b, a, K, n_key, dim)
        Y = self.p_layer_ii([Y], preprocess=False,
                            update_paras=False,
                            training=training)[0]

        return [Y[..., self.n_key_past:, :],
                Y[..., :self.n_key_past, :]]


class AlphaStructure(BaseAgentStructure):

    def __init__(self, terminal_args: list[str], manager: Structure = None, as_single_model: bool = True):
        terminal_args += ['--model_type', 'frame-based']
        super().__init__(terminal_args, manager, as_single_model)
        self.set_model_type(AlphaModel)
