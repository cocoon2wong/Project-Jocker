"""
@Author: Beihao Xia
@Date: 2023-03-20 16:15:25
@LastEditors: Conghao Wong
@LastEditTime: 2023-04-14 09:27:33
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Beihao Xia, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels import Model, layers, transformer
from codes.constant import INPUT_TYPES
from codes.training import Structure

from .__args import AgentArgs


class MinimalVModel(Model):
    """
    The `minimal` vertical model.

    - considers nothing about interactions;
    - no keypoints-interpolation two-stage subnetworks;
    - contains only the backbone;
    - considers nothing about agents' multimodality.
    """

    def __init__(self, Args: AgentArgs,
                 feature_dim: int = 128,
                 id_depth: int = 16,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        # Preprocess
        self.set_preprocess(move=0)

        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ)

        self.args = Args

        # Parameters
        self.d = feature_dim
        self.d_id = id_depth

        # Layers
        self.Tlayer, self.ITlayer = layers.get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = self.Tlayer(Oshape=(self.args.obs_frames, self.args.dim))
        self.it1 = self.ITlayer(Oshape=(self.args.pred_frames, self.args.dim))

        if type(self.t1) == layers.NoneTransformLayer:
            self.te = layers.TrajEncoding(self.d//2, tf.nn.relu, transform_layer=None)
        else:
            self.te = layers.TrajEncoding(self.d//2, tf.nn.relu, self.t1)
        self.Tsteps_en = self.t1.Tshape[0]
        self.Osteps_de = self.args.pred_frames
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        self.T = transformer.Transformer(num_layers=4,
                                         d_model=self.d,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=None,
                                         pe_input=self.Tsteps_en,
                                         pe_target=self.Tsteps_en,
                                         include_top=False)

        self.adj_fc = tf.keras.layers.Dense(self.Tsteps_de, tf.nn.tanh)
        self.gcn = layers.GraphConv(units=self.d)

        # Random id encoding
        self.ie = layers.TrajEncoding(self.d//2, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Decoder layers (with spectrums)
        self.decoder_fc1 = tf.keras.layers.Dense(2*self.d, tf.nn.tanh)
        self.decoder_fc2 = tf.keras.layers.Dense(self.Tchannels_de)

    def call(self, inputs, training=None, mask=None, *args, **kwargs):

        # unpack inputs
        trajs = inputs[0]
        bs = trajs.shape[0]

        f = self.te.call(trajs)     # (batch, obs, d/2)

        # Sample random predictions
        all_predictions = []
        rep_time = 1
        t_outputs = self.t1.call(trajs)

        for _ in range(rep_time):

            ids = tf.random.normal([bs, self.Tsteps_en, self.d_id])
            id_features = self.ie.call(ids)

            t_inputs = self.concat([f, id_features])
            t_features, _ = self.T.call(t_inputs, t_outputs, training)

            adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
            t_features = self.gcn.call(t_features, adj)

            y = self.decoder_fc1(t_features)
            y = self.decoder_fc2(y)

            y = self.it1.call(y)

            all_predictions.append(y)

        return tf.stack(all_predictions, axis=1)


class MinimalV(Structure):
    """
    Training structure for the `minimal` vertical model.
    """

    model_type = MinimalVModel

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.args = AgentArgs(terminal_args)
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

    def create_model(self, *args, **kwargs):
        return self.model_type(self.args,
                               feature_dim=self.args.feature_dim,
                               id_depth=self.args.depth,
                               structure=self,
                               *args, **kwargs)
