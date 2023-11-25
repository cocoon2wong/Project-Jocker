"""
@Author: ***
@Date: 2023-07-12 17:38:42
@LastEditors: Ziqian Zou
@LastEditTime: 2023-11-25 13:59:25
@Description: file content
@Github: ***
@Copyright 2023 ***, All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import tkinter as tk
from copy import copy
from tkinter import filedialog

import numpy as np
import torch
from matplotlib import pyplot as plt
from utils import TK_BORDER_WIDTH, TK_TITLE_STYLE, TextboxHandler

import qpid
from main import main
from qpid.constant import INPUT_TYPES
from qpid.utils import dir_check, get_mask, move_to_device

OBS = INPUT_TYPES.OBSERVED_TRAJ
NEI = INPUT_TYPES.NEIGHBOR_TRAJ

DATASET = 'ETH-UCY'
SPLIT = 'zara1'
CLIP = 'zara1'
MODEL_PATH = 'weights/evsczara1_vis'

TEMP_IMG_PATH = './temp_files/socialcircle_toy_example/fig.png'
LOG_PATH = './temp_files/socialcircle_toy_example/run.log'

dir_check(os.path.dirname(LOG_PATH))


class BetaToyExample():
    def __init__(self, args: list[str]) -> None:
        self.t: qpid.training.Structure = None
        self.image: tk.PhotoImage = None

        self.inputs: list[torch.Tensor] = None
        self.outputs: list[torch.Tensor] = None
        self.input_and_gt: list[list[torch.Tensor]] = None

        self.load_model(args)

    def init_model(self):
        self.t.create_model()
        old_input_types = self.t.agent_manager.model_inputs
        self.t.agent_manager.set_types(self.t.model.input_types,
                                       self.t.label_types)

        if self.input_and_gt is None or self.t.model.input_types != old_input_types:
            self.t.log('Reloading dataset files...')
            ds = self.t.agent_manager.clean().make(CLIP, training=False)
            self.input_and_gt = list(ds)[0]

    def load_model(self, args: list[str]):
        try:
            t = main(args, run_train_or_test=False)
            self.t = t
            self.init_model()
            self.t.log(
                f'Model `{t.args.loada}` and dataset files ({CLIP}) loaded.')
        except Exception as e:
            print(e)

    def get_input_index(self, input_type: str):
        return self.t.model.input_types.index(input_type)

    def run_on_agent(self, agent_index: int,
                     extra_neighbor_position=None):

        inputs = self.input_and_gt[0]
        inputs = [i[agent_index][None] for i in inputs]

        if (p := extra_neighbor_position) is not None:
            nei = self.add_one_neighbor(inputs, p)
            inputs[self.get_input_index(NEI)] = nei

        self.forward(inputs)
        self.draw_results()

    def add_one_neighbor(self, inputs: list[torch.Tensor],
                         position: list[tuple[float, float]]):
        '''
        Shape of `nei` should be `(1, max_agents, obs, 2)`
        '''
        obs = inputs[self.get_input_index(OBS)]
        nei = inputs[self.get_input_index(NEI)]

        nei = copy(nei.numpy())
        steps = nei.shape[-2]

        xp = np.array([0, steps-1])
        fp = np.array(position)
        x = np.arange(steps)

        traj = np.column_stack([np.interp(x, xp, fp[:, 0]),
                                np.interp(x, xp, fp[:, 1])])

        nei_count = self.get_neighbor_count(nei)
        nei[0, nei_count] = traj - obs.numpy()[0, -1:, :]
        return torch.from_numpy(nei)

    def forward(self, inputs: list[torch.Tensor]):
        self.inputs = inputs
        with torch.no_grad():
            self.outputs = self.t.model.implement(inputs, training=False)
        self.outputs = move_to_device(self.outputs, self.t.device_cpu)

    def get_neighbor_count(self, neighbor_obs: torch.Tensor):
        '''
        Input's shape should be `(1, max_agents, obs, dim)`.
        '''
        nei = neighbor_obs[0]

        if issubclass(type(nei), np.ndarray):
            nei = torch.from_numpy(nei)

        nei_mask = get_mask(torch.sum(nei, dim=[-1, -2]))
        return int(torch.sum(nei_mask))

    def draw_results(self):
        global neighbors, neighbors_traj, agent_center, predictions
        inputs = self.inputs
        outputs = self.outputs

        obs = inputs[self.get_input_index(OBS)][0].numpy()      # (obs, dim)
        nei = inputs[self.get_input_index(NEI)][0].numpy()      # (a, obs, dim)
        out = outputs[0][0].numpy()

        c_obs = self.t.picker.get_center(obs)
        c_nei = self.t.picker.get_center(nei)
        c_out = self.t.picker.get_center(out)

        plt.figure()

        # draw neighbors
        nei_count = self.get_neighbor_count(inputs[self.get_input_index(NEI)])
        _nei = c_nei[:nei_count, :, :] + c_obs[np.newaxis, -1, :]
        neighbors = _nei
        plt.plot(_nei[:, -1, 0], _nei[:, -1, 1], 'o',
                 color='darkorange', markersize=13)

        # draw neighbors' trajectories
        _nei = np.reshape(_nei, [-1, 2])
        neighbors_traj = _nei
        plt.plot(_nei[:, 0], _nei[:, 1], 's', color='purple')

        # draw observations
        agent_center = c_obs
        plt.plot(c_obs[:, 0], c_obs[:, 1], 's', color='cornflowerblue')

        # draw predictions
        predictions = c_out
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
                   canvas: tk.Canvas,
                   social_circle: tk.Label,
                   nei_angles: tk.Label):
    canvas.delete('all')
    if (px0 and py0 and px0 and py0 and
        len(x0 := px0.get()) and len(y0 := py0.get()) and
            len(x1 := px1.get()) and len(y1 := py1.get())):
        extra_neighbor = [[float(x0), float(y0)],
                          [float(x1), float(y1)]]
        t.t.log('Start running with an addition neighbor' +
                f'from {extra_neighbor[0]} to {extra_neighbor[1]}...')
    else:
        extra_neighbor = None
        t.t.log('Start running without any manual inputs...')

    t.run_on_agent(int(agent_id.get()),
                   extra_neighbor_position=extra_neighbor)
    
    # canvas.create_image(320, 240, image=t.image)
    # Draw axis X and Y
    canvas.create_line(70, 440, 570, 440)  # X
    canvas.create_line(70, 440, 70, 40)  # Y

    # X轴刻度
    for i in range(-5, 26, 1):
        x = 150 + (i * 16)
        canvas.create_line(x, 440, x, 435)
        canvas.create_text(x, 445, text=str(i), anchor=tk.N)

    # Y轴刻度
    for i in range(-5, 21, 1):
        y = 360 - (i * 16)
        canvas.create_line(70, y, 75, y)
        canvas.create_text(65, y, text=str(i), anchor=tk.E)

    # draw neighbors
    a = np.append(np.vstack(neighbors[:, -1, 0]), np.vstack(neighbors[:, -1, 1]), axis = 1).tolist()
    b = []
    for i in range(len(a)):
        b.append(tuple(a[i]))
    for x, y in b:
        canvas_x = 150 + (x * 16)
        canvas_y = 360 - (y * 16)
        canvas.create_oval(canvas_x - 3, canvas_y - 3, canvas_x + 3, canvas_y + 3, fill='orange')

    # draw neighbors' trajectories
    a = np.append(np.vstack(neighbors_traj[:, 0]), np.vstack(neighbors_traj[:, 1]), axis = 1).tolist()
    b = []
    for i in range(len(a)):
        b.append(tuple(a[i]))
    for x, y in b:
        canvas_x = 150 + (x * 16)
        canvas_y = 360 - (y * 16)
        canvas.create_oval(canvas_x - 1, canvas_y - 1, canvas_x + 1, canvas_y + 1, fill='purple')

    # draw observations
    a = np.append(np.vstack(agent_center[:, 0]), np.vstack(agent_center[:, 1]), axis = 1).tolist()
    b = []
    for i in range(len(a)):
        b.append(tuple(a[i]))
    for x, y in b:
        canvas_x = 150 + (x * 16)
        canvas_y = 360 - (y * 16)
        canvas.create_oval(canvas_x - 2, canvas_y - 2, canvas_x + 2, canvas_y + 2, fill='blue')

    # draw predictions
    for pred in predictions:
        a = np.append(np.vstack(pred[:, 0]), np.vstack(pred[:, 1]), axis = 1).tolist()
        b = []
        for i in range(len(a)):
            b.append(tuple(a[i]))
        for x, y in b:
            canvas_x = 150 + (x * 16)
            canvas_y = 360 - (y * 16)
            canvas.create_oval(canvas_x - 2, canvas_y - 2, canvas_x + 2, canvas_y + 2, fill='green')


    time = int(1000 * t.t.model.inference_times[-1])
    t.t.log(f'Running done. Time cost = {time} ms.')

    # Set numpy format
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})

    # SocialCircle
    sc = t.outputs[1][1].numpy()[0]
    social_circle.config(text=str(sc.T))

    # All neighbors' angles
    count = t.get_neighbor_count(t.inputs[t.get_input_index(NEI)])
    na = t.outputs[1][2].numpy()[0][:count]
    nei_angles.config(text=str(na*180/np.pi))

# Define transform function
def transform_coordinates(x, y):
    global x_cor, y_cor
    x_cor = round((x - 150)/16, 1)
    y_cor = round((-(y - 360)/16), 1)
    return x_cor, y_cor

# Click to set new neighbor
def on_canvas_click(event):
    global click_count, px0, py0, px1, py1
    radius = 3
    if click_count == 0:
        canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, fill='red', tags='click_point')
        x_cor, y_cor = transform_coordinates(event.x, event.y)
        px0.set(x_cor)
        py0.set(y_cor)
        click_count = 1
    elif click_count == 1:
        canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, fill='blue', tags='click_point')
        x_cor, y_cor = transform_coordinates(event.x, event.y)
        px1.set(x_cor)
        py1.set(y_cor)
        click_count = 0

# Change cursor to dot when hover on canvas
def on_canvas_move(event):
    global hover_circle
    radius = 5
    canvas.delete(hover_circle)  # 删除上一个临时圆点
    hover_circle = canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, fill='green')

# Clear canvas when click refresh button
def clear_canvas():
    canvas.delete('click_point')
    px0.set("")
    py0.set("")
    px1.set("")
    py1.set("")


if __name__ == '__main__':

    root = tk.Tk()
    root.title('Toy Example of SocialCircle Models')
    
    click_count = 0
    hover_circle = None

    # Left column
    l_args = {
        # 'background': '#FFFFFF',
        'border': TK_BORDER_WIDTH,
    }

    left_frame = tk.Frame(root, **l_args)
    left_frame.grid(row=0, column=0, sticky=tk.NW)

    tk.Label(left_frame, text='Settings',
             **TK_TITLE_STYLE, **l_args).grid(
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
        'border': TK_BORDER_WIDTH,
    }
    t_args = {
        'foreground': '#000000',
    }

    right_frame = tk.Frame(root, **r_args)
    right_frame.grid(row=0, column=1, sticky=tk.NW, rowspan=2)

    tk.Label(right_frame, text='Predictions',
             **TK_TITLE_STYLE, **r_args, **t_args).grid(
                 column=0, row=0, sticky=tk.W)

    tk.Label(right_frame, text='Model Path:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        column=0, row=1)
    (model_path := tk.Label(right_frame, width=60, wraplength=510,
                            text=MODEL_PATH, **r_args, **t_args)).grid(
        column=1, row=1)

    tk.Label(right_frame, text='Social Circle:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        column=0, row=2)
    (sc := tk.Label(right_frame, width=60, **r_args, **t_args)).grid(
        column=1, row=2)

    tk.Label(right_frame, text='Neighbor Angles:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        column=0, row=3)
    (angles := tk.Label(right_frame, width=60, **r_args, **t_args)).grid(
        column=1, row=3)

    canvas = tk.Canvas(right_frame, width=640, height=480, **r_args)
    canvas.grid(column=0, row=4, columnspan=2)
    # (canvas_label := tk.Label(right_frame, **r_args, **t_args)).grid(
    #     column=0, row=4, columnspan=2)
    
    # canvas = tk.Canvas(right_frame, width=640, height=480)
    # canvas.grid(column=0, row=4, columnspan=2)
    canvas.bind("<Button-1>", on_canvas_click)
    canvas.bind("<Motion>", on_canvas_move)

    # Log frame
    log_frame = tk.Frame(right_frame, **r_args)
    log_frame.grid(column=0, row=5, columnspan=2)

    logbar = tk.Text(log_frame, width=89, height=7, **r_args, **t_args)
    (scroll := tk.Scrollbar(log_frame, command=logbar.yview)).pack(
        side=tk.RIGHT, fill=tk.Y)
    logbar.config(yscrollcommand=scroll.set)
    logbar.pack()

    # Init model and training structure
    def args(path): return ['main.py',
                            '--sc', path,
                            '-bs', '4000',
                            '--force_dataset', DATASET,
                            '--force_split', SPLIT,
                            '--force_clip', CLIP] + sys.argv

    qpid.set_log_path(LOG_PATH)
    qpid.set_log_stream_handler(TextboxHandler(logbar))
    toy = BetaToyExample(args(MODEL_PATH))

    # Button Frame
    b_args = {
        # 'background': '#FFFFFF',
        # 'border': TK_BORDER_WIDTH,
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

    tk.Button(button_frame, text='Reload Model Weights',
              command=lambda: [toy.load_model(args(p := filedialog.askdirectory(initialdir=os.path.dirname(MODEL_PATH)))),
                               model_path.config(text=p)]).grid(
        column=0, row=12, sticky=tk.N)
    
    tk.Button(button_frame, text='Clear Canvas',
              command=lambda: clear_canvas()).grid(
        column=0, row=13, sticky=tk.N)

    root.mainloop()
