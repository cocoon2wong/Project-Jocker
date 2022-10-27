"""
@Author: Conghao Wong
@Date: 2022-06-23 10:23:53
@LastEditors: Beihao Xia
@LastEditTime: 2022-10-27 11:38:48
@Description: Second stage V^2-Net model.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import layers, transformer

from ..__args import HandlerArgs
from ..__layers import get_transform_layers
from .__baseHandler import BaseHandlerModel, BaseHandlerStructure


class VBModel(BaseHandlerModel):
    """
    Spectrum Interpolation Sub-network
    ---

    The second stage V^2-Net sub-network.
    It is used to interpolate agents' entire predictions
    by considering their interactions details.
    It also implements on agents' spectrums instead of
    their trajectories.
    """

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, feature_dim, points,
                         asHandler, key_points,
                         structure, *args, **kwargs)

        # Layers
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.linear_interpolation = layers.LinearInterpolation()

        self.Tlayer, self.ITlayer = get_transform_layers(self.args.T)

        self.t1 = self.Tlayer((self.args.obs_frames, self.args.dim))

        self.t2 = self.Tlayer(
            (self.args.obs_frames + self.args.pred_frames, self.args.dim))

        self.it = self.ITlayer(
            (self.args.obs_frames + self.args.pred_frames, self.args.dim))

        self.te = layers.TrajEncoding(units=64,
                                      activation=tf.nn.tanh,
                                      transform_layer=self.t1)

        self.ce = layers.ContextEncoding(units=64,
                                         output_channels=self.t1.Tshape[0],
                                         activation=tf.nn.tanh)

        self.transformer = transformer.Transformer(num_layers=4,
                                                   d_model=128,
                                                   num_heads=8,
                                                   dff=512,
                                                   input_vocab_size=None,
                                                   target_vocab_size=4,
                                                   pe_input=self.t1.Tshape[0],
                                                   pe_target=self.t2.Tshape[0],
                                                   include_top=True)

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None,
             *args, **kwargs):

        # unpack inputs
        trajs, maps = inputs[:2]

        traj_feature = self.te.call(trajs)
        context_feature = self.ce.call(maps)

        # transformer inputs shape = (batch, Tsteps, 128)
        t_inputs = self.concat([traj_feature, context_feature])

        # transformer target shape = (batch, obs+pred, 4)
        keypoints_index = tf.concat([[-1], keypoints_index], axis=0)
        keypoints = tf.concat([trajs[:, -1:, :], keypoints], axis=1)

        # add the last obs point to finish linear interpolation
        linear_pred = self.linear_interpolation.call(
            keypoints_index, keypoints)
        traj = tf.concat([trajs, linear_pred], axis=-2)
        t_outputs = self.t2.call(traj)

        # transformer output shape = (batch, obs+pred, 4)
        p_fft, _ = self.transformer.call(t_inputs,
                                         t_outputs,
                                         training=training)

        # decode
        p = self.it.call(p_fft)

        return p[:, self.args.obs_frames:, :]


class VB(BaseHandlerStructure):
    """
    Training structure for the second stage sub-network
    """

    def __init__(self, terminal_args: list[str], manager=None):
        super().__init__(terminal_args, manager)

        self.set_model_type(new_type=VBModel)
        self.args = HandlerArgs(terminal_args)
