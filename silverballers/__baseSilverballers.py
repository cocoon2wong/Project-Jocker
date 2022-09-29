"""
@Author: Conghao Wong
@Date: 2022-06-22 09:58:48
@LastEditors: Conghao Wong
@LastEditTime: 2022-09-29 18:58:30
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf
from codes.basemodels import Model
from codes.training import Structure

from .__args import SilverballersArgs
from .__loss import SilverballersLoss
from .agents import BaseAgentModel, BaseAgentStructure
from .handlers import BaseHandlerModel, BaseHandlerStructure, LinearHandlerModel


class BaseSilverballersModel(Model):

    def __init__(self, Args: SilverballersArgs,
                 agentModel: BaseAgentModel,
                 handlerModel: BaseHandlerModel = None,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        # processes are run in AgentModels and HandlerModels
        self.set_preprocess()

        # Layers
        self.agent = agentModel
        self.handler = handlerModel

        # Set model inputs
        a_type = self.agent.input_type
        h_type = self.handler.input_type[:-1]
        self.input_type = list(set(a_type + h_type))
        self.agent_input_index = self.get_input_index(a_type)
        self.handler_input_index = self.get_input_index(h_type)

    def get_input_index(self, input_type: list[str]):
        return [self.input_type.index(t) for t in input_type]

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None,
             *args, **kwargs):

        # call the first stage model
        agent_inputs = [inputs[i] for i in self.agent_input_index]
        agent_proposals = self.agent.forward(agent_inputs)[0]

        # call the second stage model
        handler_inputs = [inputs[i] for i in self.handler_input_index]
        handler_inputs.append(agent_proposals)
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

        # set labels and metrics
        self.set_labels('gt')
        self.set_metrics(self.Loss.avgADE, self.Loss.avgFDE)
        self.set_metrics_weights(1.0, 0.0)

        # check weights
        if 'null' in [self.args.loada, self.args.loadb]:
            raise ('`Agent` or `Handler` not found!' +
                   ' Please specific their paths via `--loada` or `--loadb`.')

        # config second-stage model
        if self.args.loadb.startswith('l'):
            handler_args = None
            handler_type = LinearHandlerModel
            handler_path = None
        else:
            handler_args = terminal_args + ['--load', self.args.loadb]
            handler_type = self.handler_model
            handler_path = self.args.loadb

        # assign substructures
        self.agent = self.substructure(self.agent_structure,
                                       args=(terminal_args +
                                             ['--load', self.args.loada]),
                                       model=self.agent_model,
                                       load=self.args.loada)

        self.handler = self.substructure(self.handler_structure,
                                         args=handler_args,
                                         model=handler_type,
                                         create_args=dict(asHandler=True),
                                         load=handler_path,
                                         key_points=self.agent.args.key_points)

        if self.args.batch_size > self.agent.args.batch_size:
            self.args._set('batch_size', self.agent.args.batch_size)

        self.args._set('dataset', self.agent.args.dataset)
        self.args._set('split', self.agent.args.split)
        self.args._set('dim', self.agent.args.dim)
        self.args._set('anntype', self.agent.args.anntype)
        self.args._set('obs_frames', self.agent.args.obs_frames)
        self.args._set('pred_frames', self.agent.args.pred_frames)

        self.add_keywords(ModelType=self.args.model,
                          PredictionType=self.args.anntype,
                          ModelName=self.args.model_name,
                          KeypointsIndex=self.agent.args.key_points,
                          AgentModelType=self.agent_model.__name__,
                          AgentModelPath=self.args.loada,
                          AgentTransformation=self.agent.args.T,
                          HandlerModelType=handler_type.__name__,
                          HandlerModelPath=handler_path,
                          HandlerTransformation=self.handler.args.T)

    def substructure(self, structure: type[BaseAgentStructure],
                     args: list[str],
                     model: type[BaseAgentModel],
                     create_args: dict = {},
                     load: str = None,
                     **kwargs):
        """
        Init a sub-structure (which contains its corresponding model).

        :param structure: class name of the training structure
        :param args: args to init the training structure
        :param model: class name of the model
        :param create_args: args to create the model, and they will be fed
            to the `structure.create_model` method
        :param load: path to load model weights
        :param **kwargs: a series of force-args that will be assigned to
            the structure's args
        """

        struct = structure(args)
        for key in kwargs.keys():
            struct.args._set(key, kwargs[key])

        struct.set_model_type(model)
        struct.model = struct.create_model(**create_args)
        struct.leader = self

        if load:
            struct.model.load_weights_from_logDir(load)

        return struct

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
            handlerModel=self.handler.model,
            structure=self,
            *args, **kwargs)

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        super().print_test_results(loss_dict, **kwargs)
        self.log(f'Test with 1st sub-network `{self.args.loada}` ' +
                 f'and 2nd seb-network `{self.args.loadb}` done.')
