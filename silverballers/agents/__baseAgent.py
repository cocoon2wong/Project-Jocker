"""
@Author: Conghao Wong
@Date: 2022-06-20 21:40:55
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-06 09:22:54
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.constant import INPUT_TYPES
from codes.managers import Model, Structure
from codes.training.loss import ADE_2D

from ..__args import AgentArgs


class BaseAgentModel(Model):

    def __init__(self, Args: AgentArgs,
                 feature_dim: int = 128,
                 id_depth: int = 16,
                 keypoints_number: int = 3,
                 keypoints_index: tf.Tensor = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.args: AgentArgs = Args
        self.structure: BaseAgentStructure = structure

        # Model input types
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ)

        # Parameters
        self.d = feature_dim
        self.d_id = id_depth

        # Keypoints and their indices
        self.n_key = keypoints_number
        self.__indices = keypoints_index
        self.num_past = tf.reduce_sum(tf.cast(self.__indices < 0, tf.int32))

        # Preprocess
        preprocess = {}
        for index, operation in enumerate(['move', 'scale', 'rotate']):
            if self.args.preprocess[index] == '1':
                preprocess[operation] = 'auto'

        self.set_preprocess(**preprocess)

    @property
    def future_keypoints_indices(self) -> tf.Tensor:
        """
        Indices of the future keypoints.
        """
        return self.__indices[self.num_past:]

    @property
    def past_keypoints_indices(self) -> tf.Tensor:
        """
        Indices of the past keypoints.
        It starts with `0`.
        """
        if self.num_past:
            return self.args.obs_frames + self.__indices[:self.num_past]
        else:
            return tf.cast([], tf.int32)

    @property
    def dim(self) -> int:
        """
        Dimension of the predicted trajectory.
        For example, `dim = 4` for 2D bounding boxes.
        """
        return self.structure.annmanager.dim

    def print_info(self, **kwargs):
        info = {'Transform type': self.args.T,
                'Index of keypoints': self.future_keypoints_indices,
                'Index of past keypoints': self.past_keypoints_indices}

        kwargs.update(**info)
        return super().print_info(**kwargs)


class BaseAgentStructure(Structure):

    model_type: BaseAgentModel = None

    def __init__(self, terminal_args: list[str],
                 manager: Structure = None,
                 is_temporary=False):

        name = 'Train Manager'
        if is_temporary:
            name += ' (First-Stage Sub-network)'

        super().__init__(args=AgentArgs(terminal_args, is_temporary),
                         manager=manager,
                         name=name)

        self.args: AgentArgs
        self.model: BaseAgentModel

        if self.args.deterministic:
            self.args._set('Kc', 1)
            self.args._set('K_train', 1)
            self.args._set('K', 1)

        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

        if self.args.loss == 'keyl2':
            self.loss.set({self.keyl2: 1.0})
        elif self.args.loss == 'avgkey':
            self.loss.set({self.avgKey: 1.0})
        else:
            raise ValueError(self.args.loss)

        self.metrics.set({self.avgKey: 1.0,
                          self.metrics.FDE: 0.0})

    def set_model_type(self, new_type: type[BaseAgentModel]):
        self.model_type = new_type

    def create_model(self) -> BaseAgentModel:
        indices = [int(i) for i in self.args.key_points.split('_')]
        indices = tf.cast(indices, tf.int32)

        return self.model_type(self.args,
                               feature_dim=self.args.feature_dim,
                               id_depth=self.args.depth,
                               keypoints_number=len(indices),
                               keypoints_index=indices,
                               structure=self)

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        s = f'python main.py --model MKII --loada {self.args.load} --loadb l'
        self.log(f'You can run `{s}` to start the silverballers evaluation.')

    def keyl2(self, outputs: list[tf.Tensor],
              labels: list[tf.Tensor],
              coe: float = 1.0,
              *args, **kwargs):
        """
        l2 loss on the future keypoints.
        Support M-dimensional trajectories.
        """
        indices = self.model.future_keypoints_indices
        labels_pickled = tf.gather(labels[0], indices, axis=-2)
        return ADE_2D(outputs[0], labels_pickled, coe=coe)

    def avgKey(self, outputs: list[tf.Tensor],
               labels: list[tf.Tensor],
               coe: float = 1.0,
               *args, **kwargs):
        """
        l2 (2D-point-wise) loss on the future keypoints.

        :param outputs: A list of tensors, where `outputs[0].shape`
            is `(batch, K, pred, 2)` or `(batch, pred, 2)`
            or `(batch, K, n_key, 2)` or `(batch, n_key, 2)`.
        :param labels: Shape of `labels[0]` is `(batch, pred, 2)`.
        """
        pred = outputs[0]
        indices = self.model.future_keypoints_indices

        if pred.ndim == 3:
            pred = pred[:, tf.newaxis, :, :]

        if pred.shape[-2] != len(indices):
            pred = tf.gather(pred, indices, axis=-2)

        labels_key = tf.gather(labels[0], indices, axis=-2)

        return self.ADE([pred], [labels_key], coe)
