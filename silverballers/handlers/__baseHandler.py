"""
@Author: Conghao Wong
@Date: 2022-06-22 09:35:52
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-14 10:42:07
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf
from codes.basemodels import Model
from codes.training import Structure

from ..__args import HandlerArgs
from ..__loss import SilverballersLoss


class BaseHandlerModel(Model):

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.structure: Structure = structure

        # GT in the inputs is only used when training
        self.set_inputs('trajs', 'maps', 'paras', 'gt')

        # Parameters
        self.asHandler = asHandler
        self.d = feature_dim
        self.points = points
        self.key_points = key_points

        if self.asHandler or key_points != 'null':
            pi = [int(i) for i in key_points.split('_')]
            self.points_index = tf.cast(pi, tf.float32)

        # Preprocess
        preprocess = {}
        for index, operation in enumerate(['move', 'scale', 'rotate']):
            if self.args.preprocess[index] == '1':
                preprocess[operation] = 'auto'

        self.set_preprocess(**preprocess)

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None):

        raise NotImplementedError

    def call_as_handler(self, inputs: list[tf.Tensor],
                        keypoints: tf.Tensor,
                        keypoints_index: tf.Tensor,
                        training=None, mask=None):
        """
        Call as the second stage handler model.
        Do NOT call this method when training.

        :param inputs: a list of trajs and context maps
        :param keypoints: predicted keypoints, shape is `(batch, K, n_k, 2)`
        :param keypoints_index: index of predicted keypoints, shape is `(n_k)`
        """

        p_all = []
        K = keypoints.shape[1]

        for k in range(K):
            # set timebar
            p = 'Calculating: {}%'.format((k+1)*100//K)
            self.structure.update_timebar(self.structure.leader.bar, p)

            # single shape is (batch, pred, 2)
            p_all.append(self.call(inputs=inputs,
                                   keypoints=keypoints[:, k, :, :],
                                   keypoints_index=keypoints_index))

        return tf.transpose(tf.stack(p_all), [1, 0, 2, 3])

    def forward(self, inputs: list[tf.Tensor],
                training=None,
                *args, **kwargs):

        keypoints = [inputs[-1]]

        inputs_p = self.process(inputs, preprocess=True, training=training)
        keypoints_p = self.process(keypoints, preprocess=True,
                                   update_paras=False,
                                   training=training)

        # only when training the single model
        if not self.asHandler:
            gt_processed = keypoints_p[0]

            if self.key_points == 'null':
                index = np.random.choice(np.arange(self.args.pred_frames-1),
                                         self.points-1)
                index = tf.concat([np.sort(index),
                                   [self.args.pred_frames-1]], axis=0)
            else:
                index = tf.cast(self.points_index, tf.int32)

            points = tf.gather(gt_processed, index, axis=1)
            index = tf.cast(index, tf.float32)

            outputs = self.call(inputs_p,
                                keypoints=points,
                                keypoints_index=index,
                                training=True)

        # use as the second stage model
        else:
            outputs = self.call_as_handler(inputs_p,
                                           keypoints=keypoints_p[0],
                                           keypoints_index=self.points_index,
                                           training=None)

        outputs_p = self.process(outputs, preprocess=False, training=training)
        return outputs_p


class BaseHandlerStructure(Structure):

    model_type = None

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.args = HandlerArgs(terminal_args)
        self.Loss = SilverballersLoss(self.args)

        self.add_keywords(NumberOfKeyoints=self.args.points,
                          Transformation=self.args.T)

        self.set_labels('gt')
        self.set_loss(self.Loss.l2)
        self.set_loss_weights(1.0)

        if self.args.key_points == 'null':
            self.set_metrics(self.Loss.avgADE,
                             self.Loss.avgFDE)
            self.set_metrics_weights(1.0, 0.0)

        else:
            self.set_metrics(self.Loss.avgADE,
                             self.Loss.avgFDE,
                             self.Loss.avgKey)
            self.set_metrics_weights(1.0, 0.0, 0.0)

    def set_model_type(self, new_type: type[BaseHandlerModel]):
        self.model_type = new_type

    def create_model(self, asHandler=False):
        return self.model_type(self.args, 128,
                               points=self.args.points,
                               asHandler=asHandler,
                               key_points=self.args.key_points,
                               structure=self)
