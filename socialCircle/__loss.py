"""
@Author: Conghao Wong
@Date: 2023-12-18 15:56:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-12-19 16:37:13
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.training.loss import BaseLossLayer


class SegLoss(BaseLossLayer):

    def forward(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                mask=None, training=None, *args, **kwargs):

        trajs = outputs[0]
        seg_maps = self.get_input(inputs, INPUT_TYPES.SEG_MAP)
        seg_map_paras = self.get_input(inputs, INPUT_TYPES.SEG_MAP_PARAS)

        batch = trajs.shape[0]
        a = seg_maps.shape[-1]

        w = seg_map_paras[..., :2]
        b = seg_map_paras[..., 2:4]

        while w.ndim < trajs.ndim:
            w = w[..., None, :]
            b = b[..., None, :]

        trajs_pixel = w * self.picker.get_center(trajs)[..., :2] + b
        trajs_pixel = torch.reshape(trajs_pixel, [batch, -1, 2])

        count = torch.arange(batch).to(trajs.device)[:, None, None]
        batch_count = count * torch.ones_like(trajs_pixel[..., :1])
        indices = torch.concat([batch_count, trajs_pixel], dim=-1)
        indices = indices.to(torch.int32).reshape([-1, 3]).T

        # Re-range all indices
        indices = torch.maximum(indices, torch.tensor(0))
        indices = torch.minimum(indices, torch.tensor(a - 1))

        seg_values = seg_maps[(indices[0], indices[1], indices[2])]
        return torch.mean(seg_values)
