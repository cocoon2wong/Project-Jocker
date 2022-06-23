"""
@Author: Conghao Wong
@Date: 2022-06-22 09:35:55
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-23 11:19:05
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import layers, transformer

from ..__args import HandlerArgs
from ..__layers import OuterLayer
from .__baseHandler import BaseHandlerModel, BaseHandlerStructure


class BurnwoodCModel(BaseHandlerModel):

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

        # Parameters
        self.steps = self.args.pred_frames

        # Layers
        self.Tlayer = layers.FFTLayer
        self.ITlayer = layers.IFFTLayer

        # Transform layers
        l = self.args.obs_frames + self.args.pred_frames
        self.t1 = self.Tlayer(Oshape=(l, 2))
        self.it1 = self.ITlayer(Oshape=(l, 2))
        
        self.linear = layers.LinearInterpolation()

        self.te = layers.TrajEncoding(units=self.d//4,
                                      activation=tf.nn.tanh,
                                      transform_layer=self.t1)

        self.ce = layers.ContextEncoding(units=self.d//4,
                                         output_channels=self.steps,
                                         activation=tf.nn.tanh)

        self.outer = OuterLayer(self.d//2, self.d//2, reshape=False)
        self.pooling = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            data_format='channels_first')

        self.outer_fc = tf.keras.layers.Dense(self.d, tf.nn.tanh)

        self.T = transformer.TransformerEncoder(num_layers=4, num_heads=8,
                                                dim_model=self.d, dim_forward=512,
                                                steps=self.steps,
                                                dim_output=4,
                                                include_top=True)

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None,
             *args, **kwargs):
        """
        Run the Burnwood model

        :param inputs: a list of tensors, which includes `trajs` and `maps`
            - trajs, shape = `(batch, obs, 2)`
            - maps, shape = `(batch, a, a)`

        :param keypoints: predicted keypoints, shape is `(batch, n_k, 2)`
        :param keypoints_index: index of predicted keypoints, shape is `(n_k)`
        :param training: controls run as the training mode or the test mode

        :return predictions: predictions, shape = `(batch, pred, 2)`
        """

        # Unpack inputs
        trajs, maps = inputs[:2]

        # Concat keypoints with the last observed point
        keypoints_index = tf.concat([[-1], keypoints_index], axis=0)
        keypoints = tf.concat([trajs[:, -1:, :], keypoints], axis=1)

        # Calculate linear interpolation and concat -> (batch, obs+pred, 2)
        # linear shape = (batch, pred, 2)
        linear = self.linear.call(keypoints_index, keypoints)
        trajs = linear

        # Encode trajectory features and context features
        traj_feature = self.te.call(trajs)      # (batch, obs+pred, d/4)
        context_feature = self.ce.call(maps)    # (batch, obs+pred, d/4)
        f = tf.concat([traj_feature, context_feature], axis=-1)

        # Outer product
        f = self.outer.call(f, f)   # (batch, obs+pred, d/4, d/4)
        f = self.pooling(f)         # (batch, obs+pred, d/8, d/8)
        f = tf.reshape(f, [f.shape[0], f.shape[1], -1])
        f = self.outer_fc(f)        # (batch, obs+pred, d)

        # Encode features with Transformer Encoder
        # (batch, obs+pred, 4)
        p_fft = self.T.call(inputs=f, training=training)
        p = self.it1.call(p_fft)

        return p


class BurnwoodC(BaseHandlerStructure):

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.set_model_type(new_type=BurnwoodCModel)

        if self.args.key_points == 'null':
            self.set_loss('ade')
            self.set_loss_weights(0.8)

        else:
            self.set_loss('ade', self.l2_keypoints)
            self.set_loss_weights(0.8, 1.0)
