"""
@Author: Conghao Wong
@Date: 2022-06-21 20:36:21
@LastEditors: Conghao Wong
@LastEditTime: 2022-08-03 16:46:34
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import cv2
import numpy as np
import tensorflow as tf

from .. import dataset
from ..args import Args
from ..utils import (DISTRIBUTION_COLORBAR, DISTRIBUTION_IMAGE, GT_IMAGE,
                     OBS_IMAGE, PRED_IMAGE)

CONV_LAYER = tf.keras.layers.Conv2D(
    1, (20, 20), (1, 1), 'same',
    kernel_initializer=tf.initializers.constant(1/(20*20)))


class Visualization():
    """
    Visualization
    -------------
    Visualize results on video datasets

    Properties
    ----------
    ```python
    >>> self.video_capture  # Video capture
    >>> self.video_paras    # a list of [sample_step, frame_rate]
    >>> self.video_weights  # weights to tansfer real scale to pixel
    ```

    Methods
    -------
    ```python
    # setup video parameters
    >>> self.set_video(video_capture, video_paras, video_weights)

    # transfer coordinates from real scale to pixels
    >>> self.real2pixel(real_pos)
    """

    def __init__(self, args: Args, dataset: str, clip: str):

        self.args = args
        self.info = dataset.VideoClip(name=clip, dataset=dataset).get()

        self._vc = None
        self._paras = None

        self._scale = self.info.datasetInfo.scale
        self._scale_vis = self.info.datasetInfo.scale_vis

        self.order = self.info.order
        self.set_video(self.info)

        self.obs_file = cv2.imread(OBS_IMAGE, -1)
        self.pred_file = cv2.imread(PRED_IMAGE, -1)
        self.gt_file = cv2.imread(GT_IMAGE, -1)
        self.dis_file = cv2.imread(DISTRIBUTION_IMAGE, -1)

    @property
    def video_capture(self) -> cv2.VideoCapture:
        return self._vc

    @property
    def video_paras(self):
        """
        [sample_step, frame_rate]
        """
        return self._paras

    @property
    def scale(self):
        """
        annotation scales
        """
        return self._scale

    @property
    def scale_vis(self):
        """
        Video scale
        """
        return self._scale_vis

    @property
    def video_matrix(self) -> list[float]:
        """
        transfer weights from real scales to pixels.
        """
        return self._matrix

    def set_video(self, video_info: dataset.VideoClip):

        path = video_info.video_path
        vc = cv2.VideoCapture(path)

        if vc.open(path):
            self._vc = vc

        try:
            self.scene_image = cv2.imread(path)
        except:
            self.scene_image = None

        self._paras = video_info.paras
        self._matrix = video_info.matrix

    def real2pixel(self, real_pos):
        """
        Transfer coordinates from real scale to pixels.

        :param real_pos: coordinates, shape = (n, 2) or (k, n, 2)
        :return pixel_pos: coordinates in pixels
        """
        scale = self.scale
        weights = self.video_matrix

        if type(real_pos) == list:
            real_pos = np.array(real_pos)

        if len(real_pos.shape) == 2:
            real_pos = real_pos[np.newaxis, :, :]

        d = self.args.dim

        all_results = []
        for step in range(real_pos.shape[1]):
            # position at one step, shape = (k, d)
            r = scale * real_pos[:, step, :]

            # both model and dataset support `boundingbox`
            if (self.info.datasetInfo.anntype == 'boundingbox' and
                    self.args.anntype == 'boundingbox'):
                result = []
                for index in range(0, self.args.dim, 2):
                    result += [weights[0] * r.T[index+self.order[0]] + weights[1],
                               weights[2] * r.T[index+self.order[1]] + weights[3]]
                result = np.column_stack(result).astype(np.int32)

            # when model only support `coordinate`
            elif self.args.anntype.startswith('coordinate'):
                result = np.column_stack([
                    weights[0] * r.T[self.order[0]] + weights[1],
                    weights[2] * r.T[self.order[1]] + weights[3],
                ]).astype(np.int32)

            all_results.append(result)

        return np.array(all_results)

    @staticmethod
    def add_png_value(source, png, position, alpha=1.0):
        yc, xc = position
        xp, yp, _ = png.shape
        xs, ys, _ = source.shape
        x0, y0 = [xc-xp//2, yc-yp//2]

        if x0 >= 0 and y0 >= 0 and x0 + xp <= xs and y0 + yp <= ys:
            source[x0:x0+xp, y0:y0+yp, :3] = \
                source[x0:x0+xp, y0:y0+yp, :3] + \
                png[:, :, :3] * alpha * png[:, :, 3:]/255

        return source

    def draw(self, agents: list[dataset.Agent],
             frame_id: Union[str, int],
             save_path='null',
             show_img=False,
             draw_distribution=False):
        """
        Draw trajecotries on images.

        :param agents: a list of agent managers (`Agent`)
        :param frame_id: name of the frame to draw on
        :param save_path: save path
        :param show_img: controls if show results in opencv window
        :draw_distrubution: controls if draw as distribution for generative models
        """
        draw = False
        if self.video_capture:
            time = 1000 * frame_id / self.video_paras[1]
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, time - 1)
            _, f = self.video_capture.read()
            draw = True

        if (f is None) and (self.scene_image is not None):
            f = self.scene_image
            draw = True

        if not draw:
            raise FileNotFoundError(
                'Video at `{}` NOT FOUND.'.format(self.info.video_path))

        for agent in agents:
            obs = self.real2pixel(agent.traj)
            pred = self.real2pixel(agent.pred)
            gt = self.real2pixel(agent.groundtruth) \
                if len(agent.groundtruth) else None
            f = self._visualization(f, obs, gt, pred,
                                    draw_distribution,
                                    alpha=1.0)

        texts = ['{}'.format(self.info.name),
                 'frame: {}'.format(str(frame_id).zfill(6)),
                 'agent: {}'.format(agent.id)]

        for index, text in enumerate(texts):
            f = cv2.putText(f, text,
                            org=(10 + 3, 40 + index * 30 + 3),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.9,
                            color=(0, 0, 0),
                            thickness=2)

            f = cv2.putText(f, text,
                            org=(10, 40 + index * 30),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.9,
                            color=(255, 255, 255),
                            thickness=2)

        if self.scale_vis > 1:
            original_shape = f.shape
            f = cv2.resize(
                f, (int(original_shape[1]/self.scale_vis),
                    int(original_shape[0]/self.scale_vis)))

        if show_img:
            cv2.namedWindow(self.info.name, cv2.WINDOW_NORMAL |
                            cv2.WINDOW_KEEPRATIO)
            f = f.astype(np.uint8)
            cv2.imshow(self.info.name, f)
            cv2.waitKey(80)

        else:
            cv2.imwrite(save_path, f)

    def draw_video(self, agent: dataset.Agent, save_path, interp=True, indexx=0, draw_distribution=False):
        _, f = self.video_capture.read()
        video_shape = (f.shape[1], f.shape[0])

        frame_list = (np.array(agent.frame_list).astype(
            np.float32)).astype(np.int32)
        frame_list_original = frame_list

        if interp:
            frame_list = np.array(
                [index for index in range(frame_list[0], frame_list[-1]+1)])

        video_list = []
        times = 1000 * frame_list / self.video_paras[1]

        obs = self.real2pixel(agent.traj)
        gt = self.real2pixel(agent.groundtruth)
        pred = self.real2pixel(agent.pred)

        # # load from npy file
        # pred = np.load('./figures/hotel_{}_stgcnn.npy'.format(indexx)).reshape([-1, 2])
        # pred = self.real2pixel(np.column_stack([
        #     pred.T[0],  # sr: 0,1; sgan: 1,0; stgcnn: 1,0
        #     pred.T[1],
        # ]), traj_weights)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        VideoWriter = cv2.VideoWriter(
            save_path, fourcc, self.video_paras[1], video_shape)

        for time, frame in zip(times, frame_list):
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC, time - 1)
            _, f = self.video_capture.read()

            # draw observations
            for obs_step in range(agent.obs_length):
                if frame >= frame_list_original[obs_step]:
                    f = ADD(
                        f, self.obs_file, obs[obs_step])

            # draw predictions
            if frame >= frame_list_original[agent.obs_length]:
                f = self._visualization(
                    f, pred=pred, draw_distribution=draw_distribution)

            # draw GTs
            for gt_step in range(agent.obs_length, agent.total_frame):
                if frame >= frame_list_original[gt_step]:
                    f = ADD(
                        f, self.gt_file, gt[gt_step - agent.obs_length])

            video_list.append(f)
            VideoWriter.write(f)

    def _visualization(self, f: np.ndarray,
                       obs=None, gt=None, pred=None,
                       draw_distribution: int = None,
                       alpha=1.0):
        """
        Draw one agent's observations, predictions, and groundtruths.

        :param f: image file
        :param obs: (optional) observations in *pixel* scale
        :param gt: (optional) ground truth in *pixel* scale
        :param pred: (optional) predictions in *pixel* scale, shape = `(steps, (K), dim)`
        :param draw_distribution: controls if draw as a distribution
        :param alpha: alpha channel coefficient
        """
        f_original = f.copy()
        f = np.zeros([f.shape[0], f.shape[1], 4])
        anntype = self.args.anntype

        # draw observations
        if obs is not None:
            f = draw_traj(f, obs, self.obs_file,
                          color=(255, 255, 255),
                          width=3, alpha=alpha,
                          anntype=anntype)

        # draw ground truth future trajectories
        if gt is not None:
            f = draw_traj(f, gt, self.gt_file,
                          color=(255, 255, 255),
                          width=3, alpha=alpha,
                          anntype=anntype)

        # add video image
        f = ADD(f_original, f,
                [f.shape[1]//2, f.shape[0]//2],
                alpha)

        # draw predicted trajectories
        if pred is not None:
            f = draw_pred(f, pred, self.pred_file,
                          width=3, alpha=alpha,
                          anntype=anntype,
                          draw_distribution=draw_distribution,
                          dis_file=self.dis_file,
                          color_bar=DISTRIBUTION_COLORBAR)

        return f


def ADD(source: np.ndarray,
        png: np.ndarray,
        position: np.ndarray,
        alpha=1.0):
    """
    Add a png file to the source image

    :param source: source image, shape = `(H, W, 3)` or `(H, W, 4)`
    :param png: png image, shape = `(H, W, 3)` or `(H, W, 4)`
    :param position: pixel-level position in the source image, shape = `(2)`
    :param alpha: transparency
    """

    yc, xc = position
    xp, yp, _ = png.shape
    xs, ys, _ = source.shape
    x0, y0 = [xc-xp//2, yc-yp//2]

    if png.shape[-1] == 4:
        png_mask = png[:, :, 3:4]/255
        png_file = png[:, :, :3]
    else:
        png_mask = np.ones_like(png)
        png_file = png

    if x0 >= 0 and y0 >= 0 and x0 + xp <= xs and y0 + yp <= ys:
        source[x0:x0+xp, y0:y0+yp, :3] = \
            (1.0 - alpha * png_mask) * source[x0:x0+xp, y0:y0+yp, :3] + \
            alpha * png_mask * png_file

        if source.shape[-1] == 4:
            source[x0:x0+xp, y0:y0+yp, 3:4] = \
                np.minimum(source[x0:x0+xp, y0:y0+yp, 3:4] +
                           255 * alpha * png_mask, 255)
    return source


def __draw_single_coordinate(source, coor: np.ndarray, png_file,
                             color, width, alpha):
    """
    shape of `coor` is `(2)`
    """
    return ADD(source, png_file, (coor[1], coor[0]), alpha)


def __draw_single_boundingbox(source, box: np.ndarray, png_file,
                              color, width, alpha):
    """
    shape of `box` is `(4)`
    """
    (y1, x1, y2, x2) = box[:4]
    cv2.rectangle(img=source,
                  pt1=(x1, y1),
                  pt2=(x2, y2),
                  color=color,
                  thickness=width)
    return source


def __draw_traj_coordinates(source, trajs, png_file,
                            color, width, alpha):
    """
    shape of `trajs` is `(steps, 2)`
    """
    trajs = np.column_stack([trajs.T[1], trajs.T[0]])

    if len(trajs) >= 2:
        for left, right in zip(trajs[:-1], trajs[1:]):
            # draw lines
            cv2.line(img=source,
                     pt1=(left[0], left[1]),
                     pt2=(right[0], right[1]),
                     color=color, thickness=width)

            # draw points
            source[:, :, 3] = alpha * 255 * source[:, :, 0]/color[0]
            source = ADD(source, png_file, left)

    # draw the last point
    source = ADD(source, png_file, trajs[-1])
    return source


def __draw_traj_boundingboxes(source, trajs, png_file,
                              color, width, alpha):
    """
    shape of `trajs` is `(steps, 4)`
    """
    for box in trajs:
        source = __draw_single_boundingbox(
            source, box, png_file, color, width, alpha)

    # draw center point
    source[:, :, 3] = alpha * 255 * source[:, :, 0]/color[0]
    for box in trajs:
        source = ADD(source, png_file,
                     ((box[1]+box[3])//2, (box[0]+box[2])//2))

    return source


def draw_traj(source, trajs, png_file,
              color=(255, 255, 255),
              width=3, alpha=1.0,
              anntype='coordinate'):
    """
    Draw lines and points.
    `color` in (B, G, R)

    :param source: a ndarray that contains the image, shape = `(H, W, 4)`
    :param trajs: trajectories, shape = `(steps, (1), dim)`
    :param png_file: a ndarray that contains the png icon, shape = `(H, W, 4)`
    :param width:
    :param alpha:
    :param anntype: annotation type, canbe `'coordinate'` or `'boundingbox'`
    """
    if (len(s := trajs.shape) == 3) and (s[1] == 1):
        trajs = trajs[:, 0, :]

    if anntype == 'coordinate':
        source = __draw_traj_coordinates(source, trajs, png_file,
                                         color, width, alpha)

    elif anntype == 'boundingbox':
        source = __draw_traj_boundingboxes(source, trajs, png_file,
                                           color, width, alpha)

    else:
        raise ValueError(anntype)

    return source


def draw_pred(source, pred: np.ndarray, png_file,
              width, alpha,
              anntype: str,
              draw_distribution: int,
              dis_file=None, color_bar=None):
    """
    shape of `pred` is `(steps, (K), dim)`
    """

    f = source
    dim = pred.shape[-1]
    background = np.zeros(f.shape[:2] + (4,))

    if draw_distribution == 1:
        f1 = draw_dis(background, pred.reshape([-1, 2]),
                      dis_file, color_bar,
                      alpha=0.5)

    elif draw_distribution == 2:
        all_steps = pred.shape[0]
        for index, step in enumerate(pred):
            f1 = draw_dis(background, step, dis_file,
                          index/all_steps * color_bar,
                          alpha=1.0)

    if draw_distribution > 0:
        f_smooth = CONV_LAYER(np.transpose(f1.astype(np.float32), [2, 0, 1])[
            :, :, :, np.newaxis]).numpy()
        f_smooth = np.transpose(f_smooth[:, :, :, 0], [1, 2, 0])
        f1 = f_smooth

        return ADD(f, f1,
                   [f.shape[1]//2, f.shape[0]//2],
                   alpha=0.8)

    else:
        if pred.ndim == 2:
            pred = pred[:, np.newaxis, :]

        if anntype == 'coordinate':
            draw_func = __draw_single_coordinate
        elif anntype == 'boundingbox':
            draw_func = __draw_single_boundingbox
        else:
            raise ValueError(anntype)

        for pred_k in np.transpose(pred, [1, 0, 2]):
            color = (np.random.randint(0, 256),
                     np.random.randint(0, 256),
                     np.random.randint(0, 256))

            for p in pred_k:
                source = draw_func(source, p, png_file,
                                   color=color,
                                   width=3, alpha=alpha)

        return source


def draw_dis(source, trajs, png_file, color_bar: np.ndarray, alpha=1.0):
    dis = np.zeros([source.shape[0], source.shape[1], 3])
    for p in trajs:
        dis = Visualization.add_png_value(dis, png_file, (p[1], p[0]), alpha)
    dis = dis[:, :, -1]

    if not dis.max() == 0:
        dis = dis ** 0.2
        alpha_channel = (255 * dis/dis.max()).astype(np.int32)
        color_map = color_bar[alpha_channel]
        distribution = np.concatenate(
            [color_map, np.expand_dims(alpha_channel, -1)], axis=-1)
        source = ADD(
            source, distribution, [source.shape[1]//2, source.shape[0]//2], alpha)

    return source


def draw_relation(source, points, png_file, color=(255, 255, 255), width=2):
    left = points[0]
    right = points[1]
    cv2.line(source, (left[0], left[1]), (right[0], right[1]), color, width)
    source = ADD(source, png_file, left)
    source = ADD(source, png_file, right)
    return source
