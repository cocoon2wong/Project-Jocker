"""
@Author: Conghao Wong
@Date: 2023-07-12 17:38:42
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-27 20:53:42
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os
import sys
import tkinter as tk
from copy import copy, deepcopy
from tkinter import filedialog
from typing import Any

import numpy as np
import torch
from utils import TK_BORDER_WIDTH, TK_TITLE_STYLE, TextboxHandler

sys.path.insert(0, os.path.abspath('.'))

import qpid
from main import main
from qpid.constant import INPUT_TYPES
from qpid.dataset.agent_based import Agent
from qpid.mods import vis
from qpid.utils import dir_check, get_mask, move_to_device

OBS = INPUT_TYPES.OBSERVED_TRAJ
NEI = INPUT_TYPES.NEIGHBOR_TRAJ

DATASET = 'ETH-UCY'
SPLIT = 'zara1'
CLIP = 'zara1'
MODEL_PATH = 'weights/Silverbullet-Torch/evsczara1_vis'

TEMP_IMG_PATH = './temp_files/socialcircle_toy_example/fig.png'
TEMP_RGB_IMG_PATH = './temp_files/socialcircle_toy_example/fig_rgb.png'
LOG_PATH = './temp_files/socialcircle_toy_example/run.log'

DRAW_MODE_PLT = 'PLT'
DRAW_MODE_QPID = 'INTERACTIVE'
DRAW_MODES_ALL = [DRAW_MODE_QPID, DRAW_MODE_PLT]

MAX_HEIGHT = 480
MAX_WIDTH = 640

MARK_CIRCLE_RADIUS = 3
MARKER_RADIUS = 5

dir_check(os.path.dirname(LOG_PATH))


class SocialCircleToy():
    def __init__(self, args: list[str]) -> None:
        # Manager objects
        self.t: qpid.training.Structure | None = None
        self.image: tk.PhotoImage | None = None
        self.vis_mgr: vis.Visualization | None = None

        # Data containers
        self.inputs: list[torch.Tensor] | None = None
        self.outputs: list[torch.Tensor] | None = None
        self.input_and_gt: list[list[torch.Tensor]] | None = None

        # Settings
        self.draw_mode_count = 0
        self.click_count = 0
        self.marker_count: int | None = None

        # Variables
        self.image_scale = 1.0
        self.image_margin = [0.0, 0.0]

        # Try to load models from the init args
        self.load_model(args)

        # TK variables
        self.tk_vars: dict[str, tk.StringVar] = {}
        self.tk_vars['agent_id'] = tk.StringVar(value='1195')
        for i in ['px0', 'py0', 'px1', 'py1']:
            self.tk_vars[i] = tk.StringVar()

    @property
    def draw_mode(self) -> str:
        return DRAW_MODES_ALL[self.draw_mode_count]

    @property
    def agents(self):
        if self.t:
            agents = self.t.agent_manager.agents
        else:
            raise ValueError(self.t)
        return agents

    def init_model(self):
        """
        Init models and managers, then load all needed data.
        """
        if not self.t:
            raise ValueError('Structure Not Initialized!')

        # Create model(s)
        self.t.create_model()
        old_input_types = self.t.agent_manager.model_inputs
        self.t.agent_manager.set_types(self.t.model.input_types,
                                       self.t.label_types)

        # Load dataset files
        if self.input_and_gt is None or self.t.model.input_types != old_input_types:
            self.t.log('Reloading dataset files...')
            ds = self.t.agent_manager.clean().make(CLIP, training=False)
            self.input_and_gt = list(ds)[0]

        # Create vis manager
        if not self.vis_mgr:
            self.vis_mgr = vis.Visualization(manager=self.t,
                                             dataset=self.t.args.force_dataset,
                                             clip=self.t.args.force_clip)

    def load_model(self, args: list[str]):
        """
        Create new models and training structures from the given args.
        """
        try:
            t = main(args, run_train_or_test=False)
            self.t = t
            self.init_model()
            self.t.log(
                f'Model `{t.args.loada}` and dataset files ({CLIP}) loaded.')
        except Exception as e:
            print(e)

    def get_input_index(self, input_type: str):
        if not self.t:
            raise ValueError
        return self.t.model.input_types.index(input_type)

    def run_on_agent(self, agent_index: int,
                     extra_neighbor_position=None):

        if not self.input_and_gt:
            raise ValueError

        inputs = self.input_and_gt[0]
        inputs = [i[agent_index][None] for i in inputs]

        if (p := extra_neighbor_position) is not None:
            nei = self.add_one_neighbor(inputs, p)
            inputs[self.get_input_index(NEI)] = nei

        self.forward(inputs)

        # Draw results on images
        m = self.draw_mode
        if m == DRAW_MODE_QPID:
            self.draw_results(agent_index, draw_with_plt=False,
                              image_save_path=TEMP_RGB_IMG_PATH,
                              resize_image=True)
        elif m == DRAW_MODE_PLT:
            self.draw_results(agent_index, draw_with_plt=True,
                              image_save_path=TEMP_IMG_PATH,
                              resize_image=False)
        else:
            raise ValueError(m)

    def get_neighbor_count(self, neighbor_obs: torch.Tensor):
        '''
        Input's shape should be `(1, max_agents, obs, dim)`.
        '''
        nei = neighbor_obs[0]

        if issubclass(type(nei), np.ndarray):
            nei = torch.from_numpy(nei)

        nei_mask = get_mask(torch.sum(nei, dim=[-1, -2]))
        return int(torch.sum(nei_mask))

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
        if not self.t:
            raise ValueError

        self.inputs = inputs
        with torch.no_grad():
            self.outputs = self.t.model.implement(inputs, training=False)
        self.outputs = move_to_device(self.outputs, self.t.device_cpu)

    def switch_draw_mode(self, label: tk.Label | None = None):
        self.draw_mode_count += 1
        self.draw_mode_count %= len(DRAW_MODES_ALL)

        if label:
            label.config(text=f'Mode: {self.draw_mode}')

    def draw_results(self, agent_index: int,
                     draw_with_plt: bool,
                     image_save_path: str,
                     resize_image=False):

        if ((not self.inputs) or
            (not self.outputs) or
            (not self.vis_mgr) or
                (not self.t)):
            raise ValueError

        # Write predicted trajectories and new neighbors to the agent
        agent = Agent().load_data(
            deepcopy(self.agents[agent_index].zip_data()))
        agent.manager = self.t.agent_manager

        agent.write_pred(self.outputs[0].numpy()[0])

        nei = self.inputs[self.get_input_index(NEI)]
        obs = self.inputs[self.get_input_index(OBS)]
        agent.neighbor_number = self.get_neighbor_count(nei)
        agent._traj_neighbor = nei[0].numpy(
        )[:agent.neighbor_number] + obs[0].numpy()[..., -1:, :]
        self.vis_mgr.draw(agent=agent,
                          frames=[agent.frames[self.t.args.obs_frames-1]],
                          save_name=image_save_path,
                          save_name_with_frame=False,
                          save_as_images=True,
                          draw_with_plt=draw_with_plt)
        del agent

        # Resize the image
        if resize_image:
            import cv2
            f = cv2.imread(image_save_path)
            h, w = f.shape[:2]
            if ((h > MAX_HEIGHT) and (h/w >= MAX_HEIGHT/MAX_WIDTH)):
                self.image_scale = h / MAX_HEIGHT
                self.image_margin = [0, (MAX_WIDTH - w/self.image_scale)//2]
            elif ((w > MAX_WIDTH) and (h/w <= MAX_HEIGHT/MAX_WIDTH)):
                self.image_scale = w / MAX_WIDTH
                self.image_margin = [(MAX_HEIGHT - h/self.image_scale)//2, 0]
            else:
                raise ValueError

            f = cv2.resize(f, [int(w//self.image_scale),
                               int(h//self.image_scale)])
            _p = os.path.join(os.path.dirname(image_save_path),
                              'resized_' + os.path.basename(image_save_path))
            cv2.imwrite(_p, f)
            image_save_path = _p

        self.image = tk.PhotoImage(file=image_save_path)

    def click(self, event: tk.Event, canvas: tk.Canvas):

        if ((not self.draw_mode == DRAW_MODE_QPID)
                or (not self.vis_mgr)):
            return

        r = MARK_CIRCLE_RADIUS

        [x, y] = [event.x, event.y]
        [px, py] = self.vis_mgr.pixel2real(
            self.image_scale * np.array([[y - self.image_margin[0],
                                          x - self.image_margin[1]]]))[0]

        if self.click_count == 0:
            canvas.delete('click_point')
            canvas.create_text(x - 2, y - 20 - 2, text='START',
                               anchor=tk.N, fill='black')
            canvas.create_text(x, y - 20, text='START',
                               anchor=tk.N, fill='white')
            canvas.create_oval(x - r, y - r, x + r, y + r,
                               fill='red', tags='click_point')
            self.click_count = 1
            self.tk_vars['px0'].set(px)
            self.tk_vars['py0'].set(py)

        elif self.click_count == 1:
            canvas.create_text(x - 2, y - 20 - 2, text='END',
                               anchor=tk.N, fill='black')
            canvas.create_text(x, y - 20, text='END',
                               anchor=tk.N, fill='white')
            canvas.create_oval(x - r, y - r, x + r, y + r,
                               fill='blue', tags='click_point')
            self.click_count = 0
            self.tk_vars['px1'].set(px)
            self.tk_vars['py1'].set(py)

        else:
            raise ValueError

    def hover(self, event: tk.Event, canvas: tk.Canvas):
        """
        Draw a dot to the canvas when hovering on it.
        """
        if not self.draw_mode == DRAW_MODE_QPID:
            return

        if self.marker_count is not None:
            canvas.delete(self.marker_count)

        self.marker_count = canvas.create_oval(event.x - MARKER_RADIUS,
                                               event.y - MARKER_RADIUS,
                                               event.x + MARKER_RADIUS,
                                               event.y + MARKER_RADIUS,
                                               fill='green')

    def run_prediction(self, with_manual_neighbor: bool,
                       canvas: tk.Canvas,
                       social_circle: tk.Label,
                       nei_angles: tk.Label):

        if self.t is None:
            raise ValueError(self.t)

        # Check if the manual neighbor exists
        if (with_manual_neighbor
            and len(x0 := self.tk_vars['px0'].get())
            and len(y0 := self.tk_vars['py0'].get())
            and len(x1 := self.tk_vars['px1'].get())
                and len(y1 := self.tk_vars['py1'].get())):

            extra_neighbor = [[float(x0), float(y0)],
                              [float(x1), float(y1)]]
            self.t.log('Start running with an addition neighbor' +
                       f'from {extra_neighbor[0]} to {extra_neighbor[1]}...')
        else:
            extra_neighbor = None
            self.t.log('Start running without any manual inputs...')

        # Run the prediction model
        self.run_on_agent(int(self.tk_vars['agent_id'].get()),
                          extra_neighbor_position=extra_neighbor)

        # Show the visualized image
        if self.image:
            canvas.create_image(MAX_WIDTH//2 + self.image_margin[1]//2,
                                MAX_HEIGHT//2 + self.image_margin[0]//2,
                                image=self.image)

        # Print model outputs
        time = int(1000 * self.t.model.inference_times[-1])
        self.t.log(f'Running done. Time cost = {time} ms.')

        # Set numpy format
        np.set_printoptions(formatter={'float': '{:0.3f}'.format})

        if (not self.outputs) or (not self.inputs):
            return

        # Print the SocialCircle
        sc = self.outputs[1][1].numpy()[0]
        social_circle.config(text=str(sc.T))

        # Print all neighbors' angles
        count = self.get_neighbor_count(self.inputs[self.get_input_index(NEI)])
        na = self.outputs[1][2].numpy()[0][:count]
        nei_angles.config(text=str(na*180/np.pi))

    def clear_canvas(self, canvas: tk.Canvas):
        """
        Clear canvas when click refresh button
        """
        canvas.delete('click_point')
        self.tk_vars['px0'].set("")
        self.tk_vars['py0'].set("")
        self.tk_vars['px1'].set("")
        self.tk_vars['py1'].set("")


if __name__ == '__main__':

    root = tk.Tk()
    root.title('Toy Example of SocialCircle Models')

    """
    Configs
    """
    # Left column
    l_args: dict[str, Any] = {
        # 'background': '#FFFFFF',
        'border': TK_BORDER_WIDTH,
    }

    # Right Column
    r_args: dict[str, Any] = {
        'background': '#FFFFFF',
        'border': TK_BORDER_WIDTH,
    }
    t_args: dict[str, Any] = {
        'foreground': '#000000',
    }

    # Button Frame
    b_args = {
        # 'background': '#FFFFFF',
        # 'border': TK_BORDER_WIDTH,
    }

    """
    Init base frames
    """
    (LF := tk.Frame(root, **l_args)).grid(
        row=0, column=0, sticky=tk.NW)
    (RF := tk.Frame(root, **r_args)).grid(
        row=0, column=1, sticky=tk.NW, rowspan=2)
    (BF := tk.Frame(root, **b_args)).grid(
        row=1, column=0, sticky=tk.N)

    """
    Init the log window
    """
    log_frame = tk.Frame(RF, **r_args)
    log_frame.grid(column=0, row=5, columnspan=2)

    logbar = tk.Text(log_frame, width=89, height=7, **r_args, **t_args)
    (scroll := tk.Scrollbar(log_frame, command=logbar.yview)).pack(
        side=tk.RIGHT, fill=tk.Y)
    logbar.config(yscrollcommand=scroll.set)
    logbar.pack()

    """
    Init the Training Structure
    """
    def args(path): return ['main.py',
                            '--sc', path,
                            '-bs', '4000',
                            '--draw_full_neighbors', '1',
                            '--force_dataset', DATASET,
                            '--force_split', SPLIT,
                            '--force_clip', CLIP] + sys.argv

    qpid.set_log_path(LOG_PATH)
    qpid.set_log_stream_handler(TextboxHandler(logbar))
    toy = SocialCircleToy(args(MODEL_PATH))

    """
    Init TK Components
    """
    tk.Label(LF, text='Settings', **TK_TITLE_STYLE, **l_args).grid(
        column=0, row=0, sticky=tk.W)

    tk.Label(LF, text='Agent ID', **l_args).grid(
        column=0, row=1)
    tk.Entry(LF, textvariable=toy.tk_vars['agent_id']).grid(
        column=0, row=2)

    tk.Label(LF, text='New Neighbor (x-axis, start)', **l_args).grid(
        column=0, row=3)
    tk.Entry(LF, textvariable=toy.tk_vars['px0']).grid(
        column=0, row=4)

    tk.Label(LF, text='New Neighbor (y-axis, start)', **l_args).grid(
        column=0, row=5)
    tk.Entry(LF,  textvariable=toy.tk_vars['py0']).grid(
        column=0, row=6)

    tk.Label(LF, text='New Neighbor (x-axis, end)', **l_args).grid(
        column=0, row=7)
    tk.Entry(LF, textvariable=toy.tk_vars['px1']).grid(
        column=0, row=8)

    tk.Label(LF, text='New Neighbor (y-axis, end)', **l_args).grid(
        column=0, row=9)
    tk.Entry(LF,  textvariable=toy.tk_vars['py1']).grid(
        column=0, row=10)

    tk.Label(RF, text='Predictions', **TK_TITLE_STYLE, **r_args, **t_args).grid(
        column=0, row=0, sticky=tk.W)

    tk.Label(RF, text='Model Path:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        column=0, row=1)
    (model_path := tk.Label(RF, width=60, wraplength=510,
                            text=MODEL_PATH, **r_args, **t_args)).grid(
        column=1, row=1)

    tk.Label(RF, text='Social Circle:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        column=0, row=2)
    (sc := tk.Label(RF, width=60, **r_args, **t_args)).grid(
        column=1, row=2)

    tk.Label(RF, text='Neighbor Angles:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        column=0, row=3)
    (angles := tk.Label(RF, width=60, **r_args, **t_args)).grid(
        column=1, row=3)

    (canvas := tk.Canvas(RF, width=MAX_WIDTH, height=MAX_HEIGHT, **r_args)).grid(
        column=0, row=4, columnspan=2)
    canvas.bind("<Motion>", lambda e: toy.hover(e, canvas))
    canvas.bind("<Button-1>", lambda e: toy.click(e, canvas))

    tk.Button(BF, text='Run Prediction',
              command=lambda: toy.run_prediction(
                  True, canvas, sc, angles), **b_args).grid(
        column=0, row=10, sticky=tk.N)

    tk.Button(BF, text='Run Prediction (original)',
              command=lambda: toy.run_prediction(
                  False, canvas, sc, angles), **b_args).grid(
        column=0, row=11, sticky=tk.N)

    tk.Button(BF, text='Reload Model Weights',
              command=lambda: [toy.load_model(args(p := filedialog.askdirectory(initialdir=os.path.dirname(MODEL_PATH)))),
                               model_path.config(text=p)]).grid(
        column=0, row=12, sticky=tk.N)

    tk.Button(BF, text='Clear Manual Inputs',
              command=lambda: toy.clear_canvas(canvas)).grid(
        column=0, row=13, sticky=tk.N)

    (mode_label := tk.Label(BF, text=f'Mode: {toy.draw_mode}', **l_args)).grid(
        column=0, row=15)

    tk.Button(BF, text='Switch Mode',
              command=lambda: toy.switch_draw_mode(mode_label)).grid(
        column=0, row=14, sticky=tk.N)

    root.mainloop()
