"""
@Author: Conghao Wong
@Date: 2022-06-20 16:27:21
@LastEditors: Conghao Wong
@LastEditTime: 2022-10-12 13:50:24
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np
import tensorflow as tf

from ..args import Args
from ..base import BaseManager
from ..basemodels import Model
from ..dataset import AgentManager, DatasetManager
from ..utils import WEIGHTS_FORMAT, dir_check
from ..vis import Visualization
from .loss import LossManager


class Structure(BaseManager):

    def __init__(self, args: list[str] = None):
        
        if issubclass(type(args), Args):
            init_args = args
        else:
            init_args = Args(args)

        super().__init__(args=init_args, manager=None)

        self.model: Model = None
        self.keywords = {}

        self.dsmanager: DatasetManager = None
        self.leader: Structure = None
        self.noTraining = False

        self.set_gpu()
        self.optimizer = self.set_optimizer()

        # Set labels, loss functions, and metrics
        # You can change the following items in your subclasses
        self.set_labels('pred')

        self.loss = LossManager('loss', manager=self)
        self.loss.set({self.loss.ADE: 1.0})

        self.metrics = LossManager('metrics', manager=self)
        if self.args.anntype == 'boundingbox':
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0,
                              self.metrics.AIoU: 0.0,
                              self.metrics.FIoU: 0.0})
        else:
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0})

        # Keywords to be print on the screen before training
        self.add_keywords(ModelType=self.args.model,
                          PredictionType=self.args.anntype,
                          ModelName=self.args.model_name)

    def set_labels(self, *args):
        """
        Set ground truths of the model
        Accept keywords:
        ```python
        groundtruth_trajectory = ['traj', 'pred', 'gt']
        destination = ['des', 'inten']

        :param input_names: type = `str`, accept several keywords
        """
        self.model_label_type = []
        for item in args:
            if 'traj' in item or \
                'gt' in item or \
                    'pred' in item:
                self.model_label_type.append('GT')

            elif 'des' in item or \
                    'inten' in item:
                self.model_label_type.append('DEST')

    def set_optimizer(self, epoch: int = None) -> tf.keras.optimizers.Optimizer:
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)
        return self.optimizer

    def set_gpu(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu.replace('_', ',')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def create_model(self) -> Model:
        """
        Create models.
        Please *rewrite* this when training new models.

        :return model: created model
        """
        raise NotImplementedError('MODEL is not defined!')

    def save_model_weights(self, save_path: str):
        """
        Save trained model to `save_path`.

        :param save_path: where model saved.
        """
        self.model.save_weights(save_path)

    def gradient_operations(self, inputs: list[tf.Tensor],
                            labels: tf.Tensor,
                            loss_move_average: tf.Variable,
                            *args, **kwargs) -> tuple[tf.Tensor, dict[str, tf.Tensor], tf.Tensor]:
        """
        Run gradient dencent once during training.

        :param inputs: model inputs
        :param labels :ground truth
        :param loss_move_average: Moving average loss

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all loss functions
        :return loss_move_average: Moving average loss
        """

        with tf.GradientTape() as tape:
            outputs = self.model.forward(inputs, training=True)
            loss, loss_dict = self.loss.call(outputs, labels,
                                             training=True,
                                             coefficient=1.0)

            loss_move_average = 0.7 * loss + 0.3 * loss_move_average

        grads = tape.gradient(loss_move_average,
                              self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_variables))

        return loss, loss_dict, loss_move_average

    def model_validate(self, inputs: list[tf.Tensor],
                       labels: tf.Tensor,
                       training=None) -> tuple[list[tf.Tensor], tf.Tensor, dict[str, tf.Tensor]]:
        """
        Run one step of forward and calculate metrics.

        :param inputs: model inputs
        :param labels :ground truth

        :return model_output: model output
        :return metrics: weighted sum of all loss 
        :return loss_dict: a dict contains all loss
        """
        outputs = self.model.forward(inputs, training)
        metrics, metrics_dict = \
            self.metrics.call(outputs, labels,
                              training=None,
                              coefficient=self.dsmanager.info.scale)

        return outputs, metrics, metrics_dict

    def train_or_test(self):
        """
        Load args, load datasets, and start training or test.
        """
        # init model
        self.model = self.create_model()

        # assign agentManagers
        self.dsmanager = DatasetManager(manager=self)
        self.dsmanager.set_types(inputs_type=self.model.input_type,
                                 labels_type=self.model_label_type)

        if self.noTraining:
            self.run_test(self.dsmanager)

        elif self.args.load == 'null':
            # restore weights before training (optional)
            if self.args.restore != 'null':
                self.model.load_weights_from_logDir(self.args.restore)

            self.log(f'Start training with args = {self.args._args_runnning}')
            self.__train()

        # prepare test
        else:
            self.log(f'Start test `{self.args.load}`')
            self.model.load_weights_from_logDir(self.args.load)
            self.run_test(self.dsmanager)

    def run_test(self, manager: DatasetManager):
        """
        Run test on the given dataset.

        :param manager: dataset's manager object
        """
        test_sets = manager.info.test_sets

        # test on a single sub-dataset
        if self.args.test_mode == 'one':
            clip = self.args.force_clip
            agents = manager.load(clip, 'test')
            self.__test(agents, self.args.dataset, [clip])

        # test on all test datasets separately
        elif self.args.test_mode == 'all':
            for clip in test_sets:
                agents = manager.load(clip, 'test')
                self.__test(agents, self.args.dataset, [clip])

        # test on all test datasets together
        elif self.args.test_mode == 'mix':
            agents = manager.load(test_sets, 'test')
            self.__test(agents, self.args.dataset, test_sets)

        else:
            raise NotImplementedError(self.args.test_mode)

    def __train(self):
        """
        Training
        """

        # print training infomation
        self.print_dataset_info(DatasetName=self.dsmanager.info.name,
                                DatasetSplitName=self.dsmanager.info.split,
                                TrainingSets=self.dsmanager.info.train_sets,
                                TestSets=self.dsmanager.info.test_sets,
                                DatasetType=self.dsmanager.info.anntype,)
        self.print_model_info()
        self.print_train_info()

        # make log directory and save current args
        self.args._save_as_json(self.args.log_dir)

        # open tensorboard
        tb = tf.summary.create_file_writer(self.args.log_dir)

        # init variables for training
        loss_move = tf.Variable(0, dtype=tf.float32)
        loss_dict = {}
        metrics_dict = {}

        best_epoch = 0
        best_metrics = 10000.0
        best_metrics_dict = {'-': best_metrics}
        test_epochs = []

        # Load dataset
        train_agents, test_agents = self.dsmanager.load('auto', 'train')
        ds_train = train_agents.make_dataset(shuffle=True)
        ds_val = test_agents.make_dataset()
        train_number = len(ds_train)

        # divide with batch size
        ds_train = ds_train.repeat(
            self.args.epochs).batch(self.args.batch_size)

        # start training
        batch_number = len(ds_train)

        epochs = []
        for batch_id, dat in enumerate(self.timebar(ds_train, text='Training...')):

            epoch = (batch_id * self.args.batch_size) // train_number

            # Update learning rate and optimizer
            if not epoch in epochs:
                self.set_optimizer(epoch)
                epochs.append(epoch)

            # Run training once
            loss, loss_dict, loss_move = self.gradient_operations(
                inputs=dat[:-1],
                labels=dat[-1],
                loss_move_average=loss_move,
                epoch=epoch,
            )

            # Check if `nan` in loss dictionary
            if tf.math.is_nan(loss):
                self.log(e := 'Find `nan` values in the loss dictionary, stop training...',
                         level='error')
                raise ValueError(e)

            # Run validation
            if ((epoch >= self.args.start_test_percent * self.args.epochs)
                    and ((epoch - 1) % self.args.test_step == 0)
                    and (not epoch in test_epochs)
                    and (epoch > 0)) or (batch_id == batch_number - 1):

                metrics, metrics_dict = self.__test_on_dataset(
                    ds=ds_val,
                    return_results=False,
                    show_timebar=False
                )
                test_epochs.append(epoch)

                # Save model
                if metrics <= best_metrics:
                    best_metrics = metrics
                    best_metrics_dict = metrics_dict
                    best_epoch = epoch

                    self.model.save_weights(os.path.join(
                        self.args.log_dir,
                        f'{self.args.model_name}_epoch{epoch}' + WEIGHTS_FORMAT
                    ))

                    np.savetxt(os.path.join(self.args.log_dir, 'best_ade_epoch.txt'),
                               np.array([best_metrics, best_epoch]))

            # Update time bar
            loss_dict = dict(epoch=epoch,
                             best=tf.stack(list(best_metrics_dict.values())),
                             **loss_dict,
                             **metrics_dict)

            for key, value in loss_dict.items():
                if issubclass(type(value), tf.Tensor):
                    loss_dict[key] = value.numpy()

            self.update_timebar(loss_dict, pos='end')

            # Write tensorboard
            with tb.as_default():
                for loss_name in loss_dict:
                    if loss_name == 'best':
                        continue

                    value = loss_dict[loss_name]
                    tf.summary.scalar(loss_name, value, step=epoch)

        self.print_train_results(best_epoch=best_epoch,
                                 best_metric=best_metrics)

    def __test(self, agents: AgentManager,
               dataset: str, clips: list[str]):
        """
        Test
        """

        # Print test information
        self.print_dataset_info(DatasetName=dataset, TestSets=clips)
        self.print_model_info()
        self.print_test_info()

        # make log directory and save current args
        if self.args.update_saved_args:
            self.args._save_as_json(self.args.load)

        # Load dataset
        ds_test = agents.make_dataset()

        # Run test
        outputs, labels, metrics, metrics_dict = self.__test_on_dataset(
            ds=ds_test,
            return_results=True,
            show_timebar=True,
        )

        # Write test results
        self.print_test_results(metrics_dict)

        # model_inputs_all = list(ds_test.as_numpy_iterator())
        outputs = stack_results(outputs)
        labels = stack_results(labels)

        self.write_test_results(outputs=outputs,
                                agents=agents,
                                clips=clips)

    def __test_on_dataset(self, ds: tf.data.Dataset,
                          return_results=False,
                          show_timebar=False):

        # init variables for test
        outputs_all = []
        labels_all = []
        metrics_all = []
        metrics_dict_all = {}

        # divide with batch size
        ds = ds.batch(self.args.batch_size)

        # hide time bar when training
        timebar = self.timebar(ds, 'Test...') if show_timebar else ds

        test_numbers = []
        for dat in timebar:
            outputs, metrics, metrics_dict = self.model_validate(
                inputs=dat[:-1],
                labels=dat[-1],
                training=False,
            )

            test_numbers.append(outputs[0].shape[0])

            if return_results:
                outputs_all = append_results_to_list(outputs, outputs_all)
                labels_all = append_results_to_list(dat[-1:], labels_all)

            # add metrics to metrics dict
            metrics_all.append(metrics)
            for key, value in metrics_dict.items():
                if not key in metrics_dict_all.keys():
                    metrics_dict_all[key] = []
                metrics_dict_all[key].append(value)

        # calculate average metric
        weights = tf.cast(tf.stack(test_numbers), tf.float32)
        metrics_all = \
            (tf.reduce_sum(tf.stack(metrics_all) * weights) /
             tf.reduce_sum(weights)).numpy()

        for key in metrics_dict_all:
            metrics_dict_all[key] = \
                (tf.reduce_sum(tf.stack(metrics_dict_all[key]) * weights) /
                 tf.reduce_sum(weights)).numpy()

        if return_results:
            return outputs_all, labels_all, metrics_all, metrics_dict_all
        else:
            return metrics_all, metrics_dict_all

    def add_keywords(self, **kwargs):
        self.keywords.update(**kwargs)

    def print_model_info(self, **kwargs):
        self.print_parameters(title='Model Options',
                              **self.keywords,
                              **kwargs)

    def print_train_info(self, **kwargs):
        self.print_parameters(title='Training Options',
                              BatchSize=self.args.batch_size,
                              GPUIndex=self.args.gpu,
                              TrainEpochs=self.args.epochs,
                              LearningRate=self.args.lr,
                              **kwargs)

    def print_test_info(self, **kwargs):
        self.print_parameters(title='Test Options',
                              BatchSize=self.args.batch_size,
                              GPUIndex=self.args.gpu,
                              **kwargs)

    def print_dataset_info(self, **kwargs):
        self.print_parameters(title='Dataset Details',
                              **kwargs)

    def print_train_results(self, best_epoch: int, best_metric: float):
        self.log('Training done.')
        self.log('During training, the model reaches the best metric ' +
                 f'`{best_metric}` at epoch {best_epoch}.')

        self.log(f'Tensorboard file is saved at `{self.args.log_dir}`. ' +
                 'To open this log file, please use `tensorboard ' +
                 f'--logdir {self.args.log_dir}`')
        self.log(f'Trained model is saved at `{self.args.log_dir}`. ' +
                 'To re-test this model, please use ' +
                 f'`python main.py --load {self.args.log_dir}`.')

    def print_test_results(self, loss_dict: dict[str, float], **kwargs):
        self.print_parameters(title='Test Results',
                              **kwargs,
                              **loss_dict)
        self.log(f'split: {self.args.split}, ' +
                 f'load: {self.args.load}, ' +
                 f'metrics: {loss_dict}')

    def write_test_results(self, outputs: list[tf.Tensor],
                           agents: AgentManager,
                           clips: list[str]):

        if (((self.args.draw_results != 'null') or
             (self.args.draw_videos != 'null'))
                and len(clips) == 1):

            # draw results on video frames
            clip = clips[0]
            tv = Visualization(self, self.args.dataset, clip)

            save_base_path = dir_check(self.args.log_dir) \
                if self.args.load == 'null' \
                else self.args.load

            img_dir = dir_check(os.path.join(save_base_path, 'VisualTrajs'))
            save_format = os.path.join(img_dir, clip + '_{}')
            self.log(f'Start saving images into `{img_dir}`...')

            pred_all = outputs[0].numpy()

            if self.args.draw_index == 'all':
                agent_indexes = list(range(len(pred_all)))
            else:
                _indexes = self.args.draw_index.split('_')
                agent_indexes = [int(i) for i in _indexes]

            for index in self.timebar(agent_indexes, 'Saving...'):
                # write traj
                agent = agents.agents[index]
                agent._traj_pred = pred_all[index]

                d = self.args.draw_distribution
                if d >= 100:
                    d -= 100
                    save_image = False
                    frames = agent.frames
                else:
                    save_image = True
                    frames = [agent.frames[self.args.obs_frames-1]]

                tv.draw(agent=agent,
                        frames=frames,
                        save_name=save_format.format(index),
                        draw_dis=d,
                        save_as_images=save_image)

            self.log(f'Prediction result images are saved at {img_dir}')


def append_results_to_list(results: list[tf.Tensor], target: list):
    if not len(target):
        [target.append([]) for _ in range(len(results))]
    [target[index].append(results[index]) for index in range(len(results))]
    return target


def stack_results(results: list[tf.Tensor]):
    for index, tensor in enumerate(results):
        results[index] = tf.concat(tensor, axis=0)
    return results
