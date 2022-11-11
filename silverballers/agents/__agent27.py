"""
@Author: Beihao Xia
@Date: 2022-10-27 19:48:35
@LastEditors: Beihao Xia
@LastEditTime: 2022-10-27 22:20:38
@Description: file content
@Github: https://github.com/conghaowoooong
@Copyright 2022 Beihao Xia, All Rights Reserved.
"""
import tensorflow as tf
from codes.basemodels import layers, transformer

from ..__args import AgentArgs
from ..__layers import get_transform_layers
from .__baseAgent import BaseAgentModel, BaseAgentStructure


class Agent27Model(BaseAgentModel):

    def __init__(self, Args: AgentArgs,
                 feature_dim: int = 128,
                 id_depth: int = 16,
                 keypoints_number: int = 3,
                 keypoints_index: tf.Tensor = None,
                 structure=None, *args, **kwargs):

        super().__init__(Args, feature_dim,
                         id_depth, keypoints_number,
                         keypoints_index, structure,
                         *args, **kwargs)
        self.Tlayer, self.ITlayer = get_transform_layers(self.args.T)

        self.tl = self.Tlayer((self.args.obs_frames, self.args.dim))
        self.itl = self.ITlayer([keypoints_number, self.args.dim])
        self.ebl = layers.TrajEncoding(self.d//2, tf.nn.relu)
        self.z_ebl = layers.TrajEncoding(self.d//2, tf.nn.tanh)

        self.trans1 = transformer.Transformer(num_layers=4,
                                              d_model=self.d,
                                              num_heads=8,
                                              dff=512,
                                              input_vocab_size=None,
                                              target_vocab_size=None,
                                              pe_input=self.tl.Tshape[0],
                                              pe_target=self.tl.Tshape[0],
                                              include_top=False)

        self.adj_fc = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.gcn = layers.GraphConv(units=self.d)

        self.d_fc1 = tf.keras.layers.Dense(self.d, tf.nn.tanh)
        self.d_fc2 = tf.keras.layers.Dense(
            self.itl.Tshape[0]*self.itl.Tshape[1])

    def call(self, inputs: list[tf.Tensor], training=None, *args, **kwargs):
        trajs = inputs[0]
        f1t = self.tl.call(trajs)  # (batch, Tsteps, Tchannels)
        f2t = self.ebl.call(f1t)  # (batch, Tsteps, d/2)

        # generate random noise z
        all_predictions = []
        rep_time = self.args.K_train if training else self.args.K

        for _ in range(rep_time):
            z = tf.random.normal(
                [trajs.shape[0], self.tl.Tshape[0], self.args.depth])
            fz = self.z_ebl.call(z)

            f2t_fz = tf.concat([f2t, fz], -1)
            trans3t, _ = self.trans1.call(f2t_fz, f1t)
            adj = tf.transpose(self.adj_fc(f2t_fz), [0, 2, 1])
            m_features = self.gcn.call(trans3t, adj)
            f4d = self.d_fc1(m_features)
            f5d = self.d_fc2(f4d)
            f6d = tf.reshape(f5d, [-1, self.args.Kc, self.itl.Tshape[0], self.itl.Tshape[1]])
            keypoints1 = self.itl.call(f6d)
            all_predictions.append(keypoints1)

        key = tf.concat(all_predictions, axis=1)

        return key


class Agent27(BaseAgentStructure):

    """
    Training structure for the `Agent47C` model.
    Note that it is only used to train the single model.
    Please use the `Silverballers` structure if you want to test any
    agent-handler based silverballers models.
    """

    def __init__(self, terminal_args: list[str], manager=None):
        super().__init__(terminal_args, manager)

        self.set_model_type(new_type=Agent27Model)
