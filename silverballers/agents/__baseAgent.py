"""
@Author: Conghao Wong
@Date: 2022-06-20 21:40:55
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-27 16:03:52
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import Model
from codes.training import Structure

from ..__args import AgentArgs
from ..__loss import SilverballersLoss


class BaseAgentModel(Model):

    def __init__(self, Args: AgentArgs,
                 feature_dim: int = 128,
                 id_depth: int = 16,
                 keypoints_number: int = 3,
                 keypoints_index: tf.Tensor = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.args = Args
        self.structure: BaseAgentStructure = structure

        # Parameters
        self.d = feature_dim
        self.d_id = id_depth
        self.n_key = keypoints_number
        self.p_index = keypoints_index

        # Preprocess
        preprocess_list = ()
        for index, operation in enumerate(['Move', 'Scale', 'Rotate']):
            if self.args.preprocess[index] == '1':
                preprocess_list += (operation,)

        self.set_preprocess(*preprocess_list)
        self.set_preprocess_parameters(move=0)


class BaseAgentStructure(Structure):

    model_type: BaseAgentModel = None

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        self.args = AgentArgs(terminal_args)
        self.Loss = SilverballersLoss(self.args)

        self.add_keywords(KeypointsIndex=self.args.key_points,
                          PreprocessOptions=self.args.preprocess,
                          Transformation=self.args.T)

        self.set_inputs('obs')
        self.set_labels('pred')

        self.set_loss(self.Loss.l2)
        self.set_loss_weights(1.0)

        self.set_metrics(self.Loss.avgKey, self.Loss.avgFDE)
        self.set_metrics_weights(1.0, 0.0)

    def set_model_type(self, new_type: type[BaseAgentModel]):
        self.model_type = new_type

    def create_model(self, *args, **kwargs):
        return self.model_type(self.args,
                               feature_dim=self.args.feature_dim,
                               id_depth=self.args.depth,
                               keypoints_number=self.Loss.p_len,
                               keypoints_index=self.Loss.p_index,
                               structure=self)

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        s = 'python main.py --model sb{} --loada {} --loadb l'
        s = s.format(self.args.model.split('agent')[-1],
                     self.args.load)

        self.log('You can run `{}` to '.format(s) +
                 'start the silverballers evaluation.')
