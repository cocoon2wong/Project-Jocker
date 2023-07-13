"""
@Author: Conghao Wong
@Date: 2023-07-12 17:38:42
@LastEditors: Conghao Wong
@LastEditTime: 2023-07-13 15:29:34
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os
import sys
import tkinter as tk

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('.'))
import codes
from codes.utils import dir_check, get_mask
from main import main

CLIP = 'zara1'
TEMP_IMG_PATH = './temp_files/beta_toy_example/fig.png'
MODEL_PATH = 'weights/Silverbullet/20230712-102506_beta_new_NEWEST_NEWESTbetazara1'


class BetaToyExample():
    def __init__(self, args: list[str]) -> None:
        self.t: codes.training.Structure = main(args, run_train_or_test=False)
        self.init_model()
        self.image: tk.PhotoImage = None

        self.inputs: list[tf.Tensor] = None
        self.outputs: list[tf.Tensor] = None

    def init_model(self):
        self.t.model = self.t.create_model()
        self.t.agent_manager.set_types(self.t.model.input_types,
                                       self.t.label_types)
        self.input_and_gt: list[list[tf.Tensor]] = \
            list(self.t.agent_manager.make(CLIP, 'test'))

    def run_on_agent(self, agent_index: int,
                     extra_neighbor_position=None):

        inputs = self.input_and_gt[agent_index][:-1]
        inputs = [i[tf.newaxis] for i in inputs]

        if (p := extra_neighbor_position) is not None:
            nei = self.add_one_neighbor(inputs, p)
            inputs[1] = nei

        self.forward(inputs)
        self.draw_results()

    def add_one_neighbor(self, inputs: list[tf.Tensor],
                         position: list[tuple[float, float]]):
        '''
        Shape of `nei` should be `(1, max_agents, obs, 2)`
        '''
        obs = inputs[0]
        nei = inputs[1]

        nei = nei.numpy()
        steps = nei.shape[-2]

        xp = np.array([0, steps-1])
        fp = np.array(position)
        x = np.arange(steps)

        traj = np.column_stack([np.interp(x, xp, fp[:, 0]),
                                np.interp(x, xp, fp[:, 1])])

        nei_count = self.get_neighbor_count(nei)
        nei[0, nei_count] = traj - obs.numpy()[0, -1:, :]
        return tf.cast(nei, tf.float32)

    def forward(self, inputs: list[tf.Tensor]):
        self.inputs = inputs
        self.outputs = self.t.model.forward(inputs, training=False)

    def get_neighbor_count(self, neighbor_obs: tf.Tensor):
        '''
        Input's shape should be `(1, max_agents, obs, dim)`.
        '''
        nei = neighbor_obs[0]
        nei_mask = get_mask(tf.reduce_sum(nei, axis=[-1, -2]))
        return int(tf.reduce_sum(nei_mask))

    def draw_results(self):
        inputs = self.inputs
        outputs = self.outputs

        obs = inputs[0][0].numpy()      # (obs, dim)
        nei = inputs[1][0].numpy()      # (max_agents, obs, dim)
        out = outputs[0][0].numpy()

        c_obs = self.t.picker.get_center(obs)
        c_nei = self.t.picker.get_center(nei)
        c_out = self.t.picker.get_center(out)

        plt.figure()
        # draw observations
        plt.plot(c_obs[:, 0], c_obs[:, 1], 'o', color='cornflowerblue')

        # draw neighbors
        nei_count = self.get_neighbor_count(inputs[1])
        _nei = c_nei[:nei_count, :, :] + c_obs[np.newaxis, -1, :]
        _nei = np.reshape(_nei, [-1, 2])
        plt.plot(_nei[:, 0], _nei[:, 1], 's', color='purple')

        # draw predictions
        for pred in c_out:
            plt.plot(pred[:, 0], pred[:, 1], 's')

        plt.axis('equal')
        save_dir = os.path.dirname(TEMP_IMG_PATH)
        dir_check(save_dir)
        plt.savefig(TEMP_IMG_PATH)
        plt.close()
        self.image = tk.PhotoImage(file=TEMP_IMG_PATH)


def run_prediction(t: BetaToyExample,
                   agent_id: tk.StringVar,
                   px0: tk.StringVar,
                   py0: tk.StringVar,
                   px1: tk.StringVar,
                   py1: tk.StringVar,
                   canvas: tk.Label,
                   social_circle: tk.Label,
                   nei_angles: tk.Label):

    if (px0 and py0 and px0 and py0 and
        len(x0 := px0.get()) and len(y0 := py0.get()) and
            len(x1 := px1.get()) and len(y1 := py1.get())):
        extra_neighbor = [[float(x0), float(y0)],
                          [float(x1), float(y1)]]
    else:
        extra_neighbor = None

    t.run_on_agent(int(agent_id.get()),
                   extra_neighbor_position=extra_neighbor)
    canvas.config(image=t.image)

    # Set numpy format
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})

    # SocialCircle
    sc = t.outputs[1][0][1].numpy()[0]
    social_circle.config(text=str(sc.T))

    # All neighbors' angles
    count = t.get_neighbor_count(t.inputs[1])
    na = t.outputs[1][0][2].numpy()[0][:count]
    nei_angles.config(text=str(na*180/np.pi))


if __name__ == '__main__':

    args = ['main.py', '--model', 'MKII',
            '--loada', MODEL_PATH,
            '--loadb', 'speed']

    toy = BetaToyExample(args)

    root = tk.Tk()
    root.title('Toy Example of SocialCircle in Beta Model')

    # Left column
    l_args = {
        # 'background': '#FFFFFF',
        'border': 5,
    }

    left_frame = tk.Frame(root, **l_args)
    left_frame.grid(row=0, column=0, sticky=tk.NW)

    tk.Label(left_frame, text='Settings',
             font=('', 24, 'bold'),
             height=2, **l_args).grid(
                 column=0, row=0, sticky=tk.W)

    agent_id = tk.StringVar(left_frame, '1195')
    tk.Label(left_frame, text='Agent ID', **l_args).grid(
        column=0, row=1)
    tk.Entry(left_frame, textvariable=agent_id).grid(
        column=0, row=2)

    px0 = tk.StringVar(left_frame)
    tk.Label(left_frame, text='New Neighbor (x-axis, start)', **l_args).grid(
        column=0, row=3)
    tk.Entry(left_frame, textvariable=px0).grid(
        column=0, row=4)

    py0 = tk.StringVar(left_frame)
    tk.Label(left_frame, text='New Neighbor (y-axis, start)', **l_args).grid(
        column=0, row=5)
    tk.Entry(left_frame,  textvariable=py0).grid(
        column=0, row=6)

    px1 = tk.StringVar(left_frame)
    tk.Label(left_frame, text='New Neighbor (x-axis, end)', **l_args).grid(
        column=0, row=7)
    tk.Entry(left_frame, textvariable=px1).grid(
        column=0, row=8)

    py1 = tk.StringVar(left_frame)
    tk.Label(left_frame, text='New Neighbor (y-axis, end)', **l_args).grid(
        column=0, row=9)
    tk.Entry(left_frame,  textvariable=py1).grid(
        column=0, row=10)

    # Right Column
    r_args = {
        'background': '#FFFFFF',
        'border': 5,
    }

    right_frame = tk.Frame(root, **r_args)
    right_frame.grid(row=0, column=1, sticky=tk.NW, rowspan=2)

    tk.Label(right_frame, text='Predictions',
             font=('', 24, 'bold'),
             height=2, **r_args).grid(column=0, row=0, sticky=tk.W)

    tk.Label(right_frame, text='Social Circle:', width=16, anchor=tk.E, **r_args).grid(
        column=0, row=1, rowspan=2)
    (sc := tk.Label(right_frame, width=60, height=3, **r_args)).grid(
        column=1, row=1, rowspan=2)

    tk.Label(right_frame, text='Neighbor Angles:', width=16, anchor=tk.E, **r_args).grid(
        column=0, row=3, rowspan=2)
    (angles := tk.Label(right_frame, width=60, height=3, **r_args)).grid(
        column=1, row=3, rowspan=2)

    tk.Canvas(right_frame, width=640, height=480, **r_args).grid(
        column=0, row=5, columnspan=2)
    (canvas := tk.Label(right_frame, **r_args)).grid(
        column=0, row=5, columnspan=2)

    # Button Frame
    b_args = {
        # 'background': '#FFFFFF',
        # 'border': 5,
    }

    button_frame = tk.Frame(root, **b_args)
    button_frame.grid(column=0, row=1, sticky=tk.N)

    tk.Button(button_frame, text='Run Prediction',
              command=lambda: run_prediction(
                  toy, agent_id, px0, py0, px1, py1,
                  canvas, sc, angles), **b_args).grid(
        column=0, row=10, sticky=tk.N)

    tk.Button(button_frame, text='Run Prediction (original)',
              command=lambda: run_prediction(
                  toy, agent_id, None, None, None, None,
                  canvas, sc, angles), **b_args).grid(
        column=0, row=11, sticky=tk.N)

    root.mainloop()
