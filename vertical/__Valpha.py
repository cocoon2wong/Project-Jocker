"""
@Author: Conghao Wong
@Date: 2022-06-23 10:13:06
@LastEditors: Conghao Wong
@LastEditTime: 2022-06-23 14:41:31
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import Model, layers, transformer
from silverballers.agents import BaseAgentStructure

from .__args import VArgs


class VAModel(Model):

    def __init__(self, Args: VArgs,
                 feature_dim: int = 128,
                 id_depth: int = None,
                 keypoints_number: int = 3,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.args = Args

        # Preprocess
        self.set_preprocess('move')
        self.set_preprocess_parameters(move=0)

        # Args
        self.n_pred = keypoints_number

        # Layers
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.te = layers.TrajEncoding(units=64,
                                      activation=tf.nn.tanh)

        self.ce = layers.ContextEncoding(units=64,
                                         output_channels=Args.obs_frames,
                                         activation=tf.nn.tanh)

        self.transformer = transformer.Transformer(num_layers=4,
                                                   d_model=128,
                                                   num_heads=8,
                                                   dff=512,
                                                   input_vocab_size=None,
                                                   target_vocab_size=None,
                                                   pe_input=Args.obs_frames,
                                                   pe_target=Args.obs_frames,
                                                   include_top=False)

        self.gcn = layers.GraphConv(units=128,
                                    activation=None)

        self.adj_fc = tf.keras.layers.Dense(Args.Kc, tf.nn.tanh)

        self.fc1 = tf.keras.layers.Dense(128, activation=tf.nn.tanh)
        self.fc2 = tf.keras.layers.Dense(self.n_pred * 2)

        self.reshape = tf.keras.layers.Reshape([Args.Kc, self.n_pred, 2])

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None) -> tf.Tensor:
        """
        Run the first stage deterministic  `Vertical-D` model

        :param inputs: a list of tensors, which includes `trajs` and `maps`
            - trajs, shape = `(batch, obs, 2)`
            - maps, shape = `(batch, a, a)`

        :param training: controls run as the training mode or the test mode

        :return predictions: predicted trajectories, shape = `(batch, Kc, N, 2)`
        """

        # unpack inputs
        trajs, maps = inputs[:2]

        traj_feature = self.te.call(trajs)
        context_feature = self.ce.call(maps)

        # transformer inputs shape = (batch, obs, 128)
        t_inputs = self.concat([traj_feature, context_feature])
        t_outputs = trajs

        # transformer
        # output shape = (batch, obs, 128)
        t_features, _ = self.transformer.call(t_inputs,
                                              t_outputs,
                                              training=training)

        adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
        # (batch, Kc, 128)
        m_features = self.gcn.call(features=t_features,
                                   adjMatrix=adj)

        # shape = (batch, Kc, 2*n)
        vec = self.fc2(self.fc1(m_features))

        # shape = (batch, Kc, n, 2)
        return self.reshape(vec)


class VA(BaseAgentStructure):
    """
    Training structure for the deterministic first stage `Vertical-D`
    """

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.args = VArgs(terminal_args)
        self.important_args = self.important_args[:-4]
        self.important_args += ['Kc', 'p_index']

        self.set_inputs('traj', 'maps')
        self.set_labels('gt')

        self.set_loss(self.l2_loss)
        self.set_loss_weights(1.0)

        self.set_metrics(self.min_FDE)
        self.set_metrics_weights(1.0)

        self.set_model_type(new_type=VAModel)
