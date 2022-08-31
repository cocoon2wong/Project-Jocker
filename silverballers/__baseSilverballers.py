"""
@Author: Conghao Wong
@Date: 2022-06-22 09:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-31 10:10:29
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import Model, layers
from codes.training import Structure

from .__args import SilverballersArgs
from .__loss import SilverballersLoss
from .agents import BaseAgentModel, BaseAgentStructure
from .handlers import BaseHandlerModel, BaseHandlerStructure


class BaseSilverballersModel(Model):

    def __init__(self, Args: SilverballersArgs,
                 agentModel: BaseAgentModel,
                 handlerModel: BaseHandlerModel = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.set_preprocess()

        self.agent = agentModel
        self.handler = handlerModel
        self.linear = not self.handler

        if self.linear:
            self.linear_layer = layers.LinearInterpolation()

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None,
             *args, **kwargs):

        outputs = self.agent.forward(inputs)

        # obtain shape parameters
        batch, Kc = outputs[0].shape[:2]
        pos = self.agent.structure.Loss.p_index

        # shape = (batch, Kc, n, 2)
        proposals = outputs[0]
        current_inputs = inputs

        if self.linear:
            # Piecewise linear interpolation
            pos = tf.cast(pos, tf.float32)
            pos = tf.concat([[-1], pos], axis=0)
            obs = current_inputs[0][:, tf.newaxis, -1:, :]
            proposals = tf.concat([tf.repeat(obs, Kc, 1), proposals], axis=-2)

            final_results = self.linear_layer.call(index=pos, value=proposals)

        else:
            # call the second stage model
            handler_inputs = [inp for inp in current_inputs]
            handler_inputs.append(proposals)
            final_results = self.handler.forward(handler_inputs)[0]

        return (final_results,)


class BaseSilverballers(Structure):

    """
    Basic structure to run the `agent-handler` based silverballers model.
    Please set agent model and handler model used in this silverballers by
    subclassing this class, and call the `set_models` method *before*
    the `super().__init__()` method.
    """

    # Structures
    agent_structure = BaseAgentStructure
    handler_structure = BaseHandlerStructure

    # Models
    agent_model = None
    handler_model = None
    silverballer_model = BaseSilverballersModel

    def __init__(self, terminal_args: list[str]):
        super().__init__(terminal_args)

        # set args
        self.args = SilverballersArgs(terminal_args)
        self.Loss = SilverballersLoss(self.args)
        self.noTraining = True

        # set inputs and outputs
        self.set_inputs('trajs', 'maps', 'paras')
        self.set_labels('gt')

        # set metrics
        self.set_metrics(self.Loss.avgADE, self.Loss.avgFDE)
        self.set_metrics_weights(1.0, 0.0)

        # check weights
        if 'null' in [self.args.loada, self.args.loadb]:
            raise ('`Agent` or `Handler` not found!' +
                   ' Please specific their paths via `--loada` or `--loadb`.')

        # assign stage-1 models
        self.agent = self.agent_structure(
            terminal_args + ['--load', self.args.loada])
        self.agent.set_model_type(self.agent_model)
        self.agent.model = self.agent.create_model()
        self.agent.model.load_weights_from_logDir(self.args.loada)
        self.agent.leader = self

        self.add_keywords(KeypointsIndex=self.agent.args.key_points,
                          AgentModelType=type(self.agent.model).__name__,
                          AgentModelPath=self.args.loada,
                          AgentTransformation=self.agent.args.T)

        if self.args.loadb.startswith('l'):
            self.linear_predict = True
            self.set_inputs('trajs')
            self.args._set('use_maps', 0)

            self.add_keywords(HandlerModelType='Linear Interpolation')

        else:
            # assign stage-2 models
            self.linear_predict = False
            self.handler = self.handler_structure(
                terminal_args + ['--load', self.args.loadb])
            self.handler.set_model_type(self.handler_model)
            self.handler.args._set('key_points', self.agent.args.key_points)
            self.handler.model = self.handler.create_model(asHandler=True)
            self.handler.model.load_weights_from_logDir(self.args.loadb)
            self.handler.leader = self

            self.add_keywords(HandlerModelType=type(self.handler.model).__name__,
                              HandlerModelPath=self.args.loadb,
                              HandlerTransformation=self.handler.args.T)

        if self.args.batch_size > self.agent.args.batch_size:
            self.args._set('batch_size', self.agent.args.batch_size)

        self.args._set('split', self.agent.args.split)
        self.args._set('dim', self.agent.args.dim)
        self.args._set('anntype', self.agent.args.anntype)
        self.args._set('obs_frames', self.agent.args.obs_frames)
        self.args._set('pred_frames', self.agent.args.pred_frames)

    def set_models(self, agentModel: type[BaseAgentModel],
                   handlerModel: type[BaseHandlerModel],
                   agentStructure: type[BaseAgentStructure] = None,
                   handlerStructure: type[BaseHandlerStructure] = None):
        """
        Set models and structures used in this silverballers instance.
        Please call this method before the `__init__` method when subclassing.
        You should better set `agentModel` and `handlerModel` rather than
        their training structures if you do not subclass these structures.
        """
        if agentModel:
            self.agent_model = agentModel

        if agentStructure:
            self.agent_structure = agentStructure

        if handlerModel:
            self.handler_model = handlerModel

        if handlerStructure:
            self.handler_structure = handlerStructure

    def create_model(self, *args, **kwargs):
        return self.silverballer_model(
            self.args,
            agentModel=self.agent.model,
            handlerModel=None if self.linear_predict else self.handler.model,
            structure=self,
            *args, **kwargs)

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        self.log(f'Test with 1st sub-network `{self.args.loada}` ' +
                 f'and 2nd seb-network `{self.args.loadb}` done.')
