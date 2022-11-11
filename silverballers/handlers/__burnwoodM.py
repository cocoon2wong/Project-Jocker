"""
@Author: Conghao Wong
@Date: 2022-11-04 14:42:45
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-04 16:39:34
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import layers, transformer

from ..__args import HandlerArgs
from ..__layers import OuterLayer, get_transform_layers, RandomMaskLayer
from .__baseHandler import BaseHandlerModel, BaseHandlerStructure


class BurnwoodMModel(BaseHandlerModel):

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, feature_dim, points,
                         asHandler, key_points, structure,
                         *args, **kwargs)

        Tlayer, ITlayer = get_transform_layers(self.args.T)

        osteps = self.args.obs_frames + self.args.pred_frames
        self.t1 = Tlayer((osteps, self.args.dim))
        self.it1 = ITlayer((osteps, self.args.dim))

        self.linear = layers.LinearInterpolation()

        self.te = layers.TrajEncoding(self.d//4, tf.nn.tanh, self.t1)
        self.ce = layers.ContextEncoding(units=self.d//4,
                                         output_channels=self.t1.Tshape[0],
                                         activation=tf.nn.tanh)
        self.mask1 = RandomMaskLayer(start_step=self.args.obs_frames,
                                     end_step=osteps-1,
                                     leave_number=points-1,
                                     all_steps=osteps)

        self.outer = OuterLayer(self.d//2, self.d//2)
        self.pooling = tf.keras.layers.MaxPooling2D(
            data_format='channels_first')
        self.fc1 = tf.keras.layers.Dense(self.d, tf.nn.tanh)

        self.T = transformer.TransformerEncoder(num_layers=4, num_heads=8,
                                                dim_model=self.d, dim_forward=512,
                                                steps=self.t1.Tshape[0],
                                                dim_output=self.t1.Tshape[1],
                                                include_top=True)

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None,
             *args, **kwargs):

        # Unpack inputs
        trajs, maps = inputs[:2]

        # Calculate linear interpolation
        keypoints_index = tf.concat([[-1], keypoints_index], axis=0)
        keypoints = tf.concat([trajs[:, -1:, :], keypoints], axis=1)
        linear = self.linear.call(keypoints_index, keypoints)

        traj_linear = tf.concat([trajs, linear], axis=-2)
        traj_linear_mask = self.mask1.call(traj_linear)

        traj_feature = self.te.call(traj_linear_mask)
        context_feature = self.ce.call(maps)
        f = tf.concat([traj_feature, context_feature], axis=-1)

        # Outer product
        f = self.outer.call(f, f)   # (batch, obs+pred, d/4, d/4)
        f = self.pooling(f)         # (batch, obs+pred, d/8, d/8)
        f = tf.reshape(f, [f.shape[0], f.shape[1], -1])
        f = self.fc1(f)        # (batch, obs+pred, d)

        # Encode features with Transformer Encoder
        # (batch, obs+pred, 4)
        p_fft = self.T.call(inputs=f, training=training)
        p = self.it1.call(p_fft)

        return p[:, self.args.obs_frames:, :]


class BurnwoodM(BaseHandlerStructure):

    def __init__(self, terminal_args: list[str], manager=None):
        super().__init__(terminal_args, manager)

        self.set_model_type(BurnwoodMModel)
