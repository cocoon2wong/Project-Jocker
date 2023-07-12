"""
@Author: Conghao Wong
@Date: 2023-07-12 17:38:42
@LastEditors: Conghao Wong
@LastEditTime: 2023-07-12 20:59:19
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
                         position: tuple[float, float]):
        """
        Shape of `nei` should be `(1, max_agents, obs, 2)`
        """
        obs = inputs[0]
        nei = inputs[1]

        nei_mask = get_mask(tf.reduce_sum(nei, axis=[-1, -2]))
        nei_count = int(tf.reduce_sum(nei_mask))

        nei = nei.numpy()
        steps = nei.shape[-2]
        ref_seq = np.ones(steps)
        traj = np.column_stack([position[0] * ref_seq,
                                position[1] * ref_seq])
        nei[0, nei_count] = traj - obs.numpy()[-1:, :]
        return tf.cast(nei, tf.float32)

    def forward(self, inputs: list[tf.Tensor]):
        self.inputs = inputs
        self.outputs = self.t.model.forward(inputs, training=False)

    def draw_results(self):
        inputs = self.inputs
        outputs = self.outputs

        obs = inputs[0][0].numpy()      # (obs, dim)
        nei = inputs[1][0].numpy()      # (max_agents, obs, dim)
        out = outputs[0][0].numpy()

        nei_mask = get_mask(tf.reduce_sum(nei, axis=[-1, -2]))
        nei_count = int(tf.reduce_sum(nei_mask))

        c_obs = self.t.picker.get_center(obs)
        c_nei = self.t.picker.get_center(nei)
        c_out = self.t.picker.get_center(out)

        plt.figure()
        # draw observations
        plt.plot(c_obs[:, 0], c_obs[:, 1], 'o', color='cornflowerblue')

        # draw neighbors
        _nei = c_nei[:nei_count, -1, :] + c_obs[-1, :]
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
                   px: tk.StringVar,
                   py: tk.StringVar,
                   canvas: tk.Label):

    if px and py and len(x := px.get()) and len(y := py.get()):
        x = float(x)
        y = float(y)
        extra_neighbor = [x, y]
    else:
        extra_neighbor = None

    t.run_on_agent(int(agent_id.get()),
                   extra_neighbor_position=extra_neighbor)
    canvas.config(image=t.image)


if __name__ == '__main__':

    args = ["main.py", "--model", "MKII",
            "--loada", MODEL_PATH,
            "--loadb", "speed"]

    toy = BetaToyExample(args)

    root = tk.Tk()
    root.title('Toy Example of SocialCircle in Beta Model')

    agent_id = tk.StringVar(root, '1195')
    tk.Label(root, text='Agent ID').grid(column=0, row=0)
    tk.Entry(root, textvariable=agent_id).grid(column=0, row=1)

    px = tk.StringVar(root)
    tk.Label(root, text='New Neighbor (x-axis)').grid(column=1, row=0)
    tk.Entry(root, textvariable=px).grid(column=1, row=1)

    py = tk.StringVar(root)
    tk.Label(root, text='New Neighbor (y-axis)').grid(column=2, row=0)
    tk.Entry(root,  textvariable=py).grid(column=2, row=1)

    (canvas := tk.Label(root)).grid(column=0, row=4, columnspan=3)

    tk.Button(root, text='Run Prediction', 
              command=lambda: run_prediction(toy, agent_id, px, py, canvas)).grid(
              column=0, row=2, columnspan=3)
    
    tk.Button(root, text='Run Prediction (Ignore the given neighbors)', 
              command=lambda: run_prediction(toy, agent_id, None, None, canvas)).grid(
              column=0, row=3, columnspan=3)

    root.mainloop()
