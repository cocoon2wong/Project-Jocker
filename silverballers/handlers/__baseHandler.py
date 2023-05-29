"""
@Author: Conghao Wong
@Date: 2022-06-22 09:35:52
@LastEditors: Beihao Xia
@LastEditTime: 2023-05-29 10:52:21
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf

from codes.constant import ANN_TYPES, INPUT_TYPES
from codes.dataset.trajectories import Annotation
from codes.managers import Model, SecondaryBar, Structure
from codes.utils import POOLING_BEFORE_SAVING

from ..__args import HandlerArgs, SilverballersArgs


class BaseHandlerModel(Model):

    is_interp_handler = False

    def __init__(self, Args: HandlerArgs,
                 feature_dim: int,
                 points: int,
                 asHandler=False,
                 key_points: str = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.args: HandlerArgs = Args
        self.structure: BaseHandlerStructure = structure

        # GT in the inputs is only used when training
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.MAP,
                        INPUT_TYPES.MAP_PARAS,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Parameters
        self.asHandler = asHandler
        self.d = feature_dim
        self.points = points
        self.key_points = key_points
        self.accept_batchK_inputs = False

        self.ext_traj_wise_outputs[1] = 'Interaction Scores'

        if self.asHandler or key_points != 'null':
            pi = [int(i) for i in key_points.split('_')]
            self.points_index = tf.cast(pi, tf.float32)

        # Preprocess
        preprocess = {}
        for index, operation in enumerate(['move', 'scale', 'rotate']):
            if self.args.preprocess[index] == '1':
                preprocess[operation] = 'auto'

        self.set_preprocess(**preprocess)

        if POOLING_BEFORE_SAVING:
            self._upsampling = tf.keras.layers.UpSampling2D(
                size=[5, 5], data_format='channels_last')

    @property
    def dim(self) -> int:
        """
        Dimension of the predicted trajectory.
        For example, `dim = 4` for 2D bounding boxes.
        """
        return self.structure.annmanager.dim

    @property
    def map_picker(self) -> Annotation:
        """
        Picker object to fix all map-related fuctions' input
        dimensions to `2` (2D center coordinate point).
        If there is no need to fix these inputs, it will return
        `None`.
        """
        # Play as a single model
        if not self.asHandler:
            if self.args.anntype != ANN_TYPES.CO_2D:
                return self.structure.picker

        # Play as the second-stage model
        else:
            if self.structure.manager.model.CO2BB:
                return None

            if self.structure.manager.args.anntype != ANN_TYPES.CO_2D:
                return self.structure.get_manager(Structure).picker

        return None

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

        if not self.accept_batchK_inputs:
            p_all = []
            for k in SecondaryBar(range(keypoints.shape[1]),
                                  manager=self.structure.manager,
                                  desc='Running Stage-2 Sub-Network...',
                                  update_main_bar=True):

                # Run stage-2 network on a batch of inputs
                pred = self.call(inputs=inputs,
                                 keypoints=keypoints[:, k, :, :],
                                 keypoints_index=keypoints_index)

                if type(pred) not in [list, tuple]:
                    pred = [pred]

                # A single output shape is (batch, pred, dim).
                p_all.append(pred[0])

            return tf.transpose(tf.stack(p_all), [1, 0, 2, 3])

        else:
            return self.call(inputs=inputs,
                             keypoints=keypoints,
                             keypoints_index=keypoints_index)

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
                index = np.arange(self.args.pred_frames-1)
                np.random.shuffle(index)
                index = tf.concat([np.sort(index[:self.points-1]),
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

        # Calculate scores
        if ((INPUT_TYPES.MAP in self.input_types) and
                (INPUT_TYPES.MAP_PARAS in self.input_types)):

            pred = outputs_p[0]
            centers = inputs[0][..., -1, :]

            if p := self.map_picker:
                pred = p.get_center(pred)[..., :2]
                centers = p.get_center(centers)[..., :2]

            scores = self.score(trajs=pred,
                                maps=inputs[1],
                                map_paras=inputs[2],
                                centers=centers)

            pred_o = outputs_p[0]

            # Pick trajectories
            if self.asHandler:
                run_args: SilverballersArgs = self.structure.manager.args
                if (p := run_args.pick_trajectories) < 1.0 and scores.ndim >= 2:
                    bs = tf.shape(scores)[0]
                    _index = tf.argsort(scores, axis=-1, direction='ASCENDING')
                    _index = _index[..., :int(p * scores.shape[-1])]

                    # gather trajectories
                    _index = _index[..., tf.newaxis]
                    count = tf.range(bs)

                    while count.ndim < _index.ndim:
                        count = count[:, tf.newaxis]

                    count = count * tf.ones_like(_index)
                    new_index = tf.concat([count, _index], axis=-1)
                    pred_o = tf.gather_nd(pred_o, new_index)

            return (pred_o, scores) + outputs_p[1:]

        else:
            return outputs_p

    def score(self, trajs: tf.Tensor,
              maps: tf.Tensor,
              map_paras: tf.Tensor,
              centers: tf.Tensor):
        """
        Calculate the score of the predicted trajectory in the
        social and scene interaction case.

        :param trajs: Predicted trajectory, shape = `(batch, pred, 2)`.
        :param maps: Trajectory map, shape = `(batch, a, a)`.
        :param map_paras: Parameters of trajectory maps, shape = `(batch, 4)`.
        :param centers: Centers of the trajectory map in the real scale. \
            It is usually the last observed point of the 2D trajectory. \
            Shape = `(batch, 1, 2)`. 
        """
        if POOLING_BEFORE_SAVING:
            maps = self._upsampling(maps[..., tf.newaxis])[..., 0]

        W = map_paras[:, :2]
        b = map_paras[:, 2:]

        while W.ndim != trajs.ndim:
            W = W[:, tf.newaxis, :]
            b = b[:, tf.newaxis, :]

        while centers.ndim != trajs.ndim:
            centers = centers[:, tf.newaxis, :]

        trajs_global_grid = (trajs - b) * W
        centers_global_grid = (centers - b) * W
        bias_grid = trajs_global_grid - centers_global_grid
        bias_grid = tf.cast(bias_grid, tf.int32)

        s = tf.shape(maps)
        map_center = tf.cast([s[-2]//2, s[-1]//2], tf.int32)
        trajs_grid = map_center[tf.newaxis] + bias_grid
        trajs_grid = tf.minimum(tf.maximum(trajs_grid, 0), s[-2]-1)

        count = tf.range(s[0])
        while count.ndim != trajs_grid.ndim:
            count = count[:, tf.newaxis]

        agent_count = count * tf.ones_like(trajs_grid[..., :1])
        index = tf.concat([agent_count, trajs_grid], axis=-1)

        all_scores = tf.gather_nd(maps, index)
        avg_scores = tf.reduce_sum(all_scores, axis=-1)

        return avg_scores

    def print_info(self, **kwargs):
        info = {'Transform type': self.args.T,
                'Number of keypoints': self.args.points}

        kwargs.update(**info)
        return super().print_info(**kwargs)


class BaseHandlerStructure(Structure):

    model_type = None

    def __init__(self, terminal_args: list[str],
                 manager: Structure = None,
                 is_temporary=False):

        name = 'Train Manager'
        if is_temporary:
            name += ' (Second-Stage Sub-network)'

        super().__init__(args=HandlerArgs(terminal_args, is_temporary),
                         manager=manager,
                         name=name)

        self.args: HandlerArgs
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)
        self.loss.set({self.loss.l2: 1.0})

        if self.args.key_points == 'null':
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0})
        else:
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0,
                              self.metrics.avgKey: 0.0})

    def set_model_type(self, new_type: type[BaseHandlerModel]):
        self.model_type = new_type

    def create_model(self, asHandler=False):
        return self.model_type(self.args,
                               feature_dim=self.args.feature_dim,
                               points=self.args.points,
                               asHandler=asHandler,
                               key_points=self.args.key_points,
                               structure=self)
