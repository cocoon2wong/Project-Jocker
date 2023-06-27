"""
@Author: Conghao Wong
@Date: 2023-06-20 16:48:45
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-27 20:56:04
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

        # Preprocess methods are applied within `call`
        self.set_preprocess()

        preprocess = {}
        for index, operation in enumerate(['move', 'scale', 'rotate']):
            if self.args.preprocess[index] == '1':
                preprocess[operation] = 'auto'

        self.process_relative = self.set_preprocess(
            builtin=False, **preprocess)
        self.process_absolute = self.set_preprocess_layers(
            builtin=False, layers=[CenterMove()],
        )

        # Layers
        tlayer, itlayer = layers.get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((self.n_key, self.dim))

        # Trajectory encoding
        self.te = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                      transform_layer=self.t1)

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

        # For social interactions
        self.te_s = layers.TrajEncoding(self.d//2, tf.nn.relu)
        self.outer_s = OuterLayer(self.d//2, self.d//2)
        self.pooling_s = layers.MaxPooling2D(
            (2, 2), data_format='channels_first')
        self.flatten_s = layers.Flatten(axes_num=2)
        self.outer_fc_s = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

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
        obs = inputs[0]         # (b:=batch, a:=agents, obs, dim)
        obs = obs * mask

        obs_absolute = self.process_absolute([obs], preprocess=True,
                                             update_paras=True,
                                             training=training)[0]

        obs_relative = self.process_relative([obs_absolute], preprocess=True,
                                             update_paras=True,
                                             training=training)[0]

        # Trajectory (relative) embedding and encoding
        f = self.te(obs_relative)
        f = self.outer(f, f)
        f = self.pooling(f)
        f = self.flatten(f)
        f_traj = self.outer_fc(f)   # (batch, agents, steps, d/2)

        if 'MIN' not in self.args.model_name:
            # Trajecotry (absolute) embedding and encoding
            obs_s = self.t1(obs_absolute)
            f_s = self.te_s(obs_s)
            f_s = self.outer_s(f_s, f_s)
            f_s = self.pooling_s(f_s)
            f_s = self.flatten_s(f_s)
            f_traj_abs = self.outer_fc_s(f_s)   # (batch, agents, steps, d/2)

            # Social embedding end encoding (max pooling)
            # Final shape is (batch, agents, steps, d/2)
            f_social = tf.reduce_max(f_traj_abs, axis=-3, keepdims=True)
            f_social = tf.repeat(f_social, self.args.max_agents, axis=-3)

            # Feature fusion
            # (batch, agents, steps, d//2)
            f_behavior = tf.concat([f_traj, f_social], axis=-1)
            f_behavior = self.concat_fc(f_behavior)
        else:
            f_behavior = f_traj

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K
        for _ in range(repeats):
            z = tf.random.normal(list(tf.shape(f_behavior)[:-1]) + [self.d_id])
            f_z = self.ie(z)
            # (b, a, steps, 2*d)
            f_final = tf.concat([f_behavior, f_z], axis=-1)

            # Transformer outputs' shape is (b, a, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=self.t1(obs_relative),
                               training=training)

            # Multiple generations
            adj = self.ms_fc(f_final)      # (b, a, steps, Kc)
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

        Y = tf.concat(all_predictions, axis=-3)   # (b, a, K, n_key, dim)
        Y = self.process_relative([Y], preprocess=False,
                                  update_paras=False,
                                  training=training)[0]
        Y = self.process_absolute([Y], preprocess=False,
                                  update_paras=False,
                                  training=training)[0]

        return [Y[..., self.n_key_past:, :],
                Y[..., :self.n_key_past, :]]


class AlphaStructure(BaseAgentStructure):

    def __init__(self, terminal_args: list[str], manager: Structure = None, as_single_model: bool = True):
        terminal_args += ['--model_type', 'frame-based']
        super().__init__(terminal_args, manager, as_single_model)
        self.set_model_type(AlphaModel)
