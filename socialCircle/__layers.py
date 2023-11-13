"""
@Author: Conghao Wong
@Date: 2023-08-08 14:55:56
@LastEditors: Conghao Wong
@LastEditTime: 2023-11-13 19:43:54
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.mods.segMaps.settings import NORMALIZED_SIZE
from qpid.utils import get_mask

INF = 1000000000
SAFE_THRESHOLDS = 0.05
MU = 0.00000001


class SocialCircleLayer(torch.nn.Module):

    def __init__(self, partitions: int,
                 max_partitions: int,
                 use_velocity: bool | int = True,
                 use_distance: bool | int = True,
                 use_direction: bool | int = True,
                 use_move_direction: bool | int = False,
                 mu=0.0001,
                 relative_velocity: bool | int = False,
                 *args, **kwargs):
        """
        A layer to compute the SocialCircle Meta-components.

        ## Partition Settings
        :param partitions: The number of partitions in the circle.
        :param max_partitions: The number of partitions (after zero padding).

        ## SocialCircle Factors
        :param use_velocity: Choose whether to use the velocity factor.
        :param use_distance: Choose whether to use the distance factor.
        :param use_direction: Choose whether to use the direction factor.
        :param use_move_direction: Choose whether to use the move direction factor.

        ## SocialCircle Options
        :param relative_velocity: Choose whether to use relative velocity or not.
        :param mu: The small number to prevent dividing zero when computing. \
            It only works when `relative_velocity` is set to `True`.
        """
        super().__init__(*args, **kwargs)

        self.partitions = partitions
        self.max_partitions = max_partitions

        self.use_velocity = use_velocity
        self.use_distance = use_distance
        self.use_direction = use_direction

        self.rel_velocity = relative_velocity
        self.use_move_direction = use_move_direction
        self.mu = mu

    @property
    def dim(self) -> int:
        """
        The number of SocialCircle factors.
        """
        return int(self.use_velocity) + int(self.use_distance) + \
            int(self.use_direction) + int(self.use_move_direction)

    def forward(self, trajs, nei_trajs, *args, **kwargs):
        # Move vectors -> (batch, ..., 2)
        # `nei_trajs` are relative values to target agents' last obs step
        obs_vector = trajs[..., -1:, :] - trajs[..., 0:1, :]
        nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]
        nei_posion_vector = nei_trajs[..., -1, :]

        # Velocity factor
        if self.use_velocity:
            # Calculate velocities
            nei_velocity = torch.norm(nei_vector, dim=-1)    # (batch, n)
            obs_velocity = torch.norm(obs_vector, dim=-1)    # (batch, 1)

            # Speed factor in the SocialCircle
            if self.rel_velocity:
                f_velocity = (nei_velocity + self.mu)/(obs_velocity + self.mu)
            else:
                f_velocity = nei_velocity

        # Distance factor
        if self.use_distance:
            f_distance = torch.norm(nei_posion_vector, dim=-1)

        # Move direction factor
        if self.use_move_direction:
            obs_move_direction = torch.atan2(obs_vector[..., 0],
                                             obs_vector[..., 1])
            nei_move_direction = torch.atan2(nei_vector[..., 0],
                                             nei_vector[..., 1])
            delta_move_direction = nei_move_direction - obs_move_direction
            f_move_direction = delta_move_direction % (2*np.pi)

        # Direction factor
        f_direction = torch.atan2(nei_posion_vector[..., 0],
                                  nei_posion_vector[..., 1])
        f_direction = f_direction % (2*np.pi)

        # Angles (the independent variable \theta)
        angle_indices = f_direction / (2*np.pi/self.partitions)
        angle_indices = angle_indices.to(torch.int32)

        # Mask neighbors
        nei_mask = get_mask(torch.sum(nei_trajs, dim=[-1, -2]), torch.int32)
        angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

        # Compute the SocialCircle
        social_circle = []
        for ang in range(self.partitions):
            _mask = (angle_indices == ang).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1)

            n = _mask_count + 0.0001
            social_circle.append([])

            if self.use_velocity:
                _velocity = torch.sum(f_velocity * _mask, dim=-1) / n
                social_circle[-1].append(_velocity)

            if self.use_distance:
                _distance = torch.sum(f_distance * _mask, dim=-1) / n
                social_circle[-1].append(_distance)

            if self.use_direction:
                _direction = torch.sum(f_direction * _mask, dim=-1) / n
                social_circle[-1].append(_direction)

            if self.use_move_direction:
                _move_d = torch.sum(f_move_direction * _mask, dim=-1) / n
                social_circle[-1].append(_move_d)

        # Shape of the final SocialCircle: (batch, p, 3)
        social_circle = [torch.stack(i) for i in social_circle]
        social_circle = torch.stack(social_circle)
        social_circle = torch.permute(social_circle, [2, 0, 1])

        if (((m := self.max_partitions) is not None) and
                (m > (n := self.partitions))):
            paddings = [0, 0, 0, m - n, 0, 0]
            social_circle = torch.nn.functional.pad(social_circle, paddings)

        return social_circle, f_direction


class PhysicalCircleLayer(torch.nn.Module):

    def __init__(self, partitions: int,
                 max_partitions: int,
                 vision_radius: str,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.partitions = partitions
        self.max_partitions = max_partitions

        self.radius = [float(r) for r in vision_radius.split('_') if len(r)]

        # Compute all pixels' indices
        xs, ys = torch.meshgrid(torch.arange(NORMALIZED_SIZE),
                                torch.arange(NORMALIZED_SIZE),
                                indexing='ij')
        self.map_pos_pixel = torch.stack(
            [xs.reshape([-1]), ys.reshape([-1])], dim=-1).to(torch.float32)

    @property
    def dim(self) -> int:
        """
        The number of PhysicalCircle factors.
        """
        return len(self.radius)

    def forward(self, seg_maps: torch.Tensor,
                seg_map_paras: torch.Tensor,
                trajectories: torch.Tensor,
                current_pos: torch.Tensor,
                *args, **kwargs):

        # Move back to original trajectories
        _obs = trajectories + current_pos

        # Treat seg maps as a long sequence
        _maps = torch.flatten(seg_maps, start_dim=1, end_dim=-1)
        map_safe_mask = (_maps <= SAFE_THRESHOLDS).to(torch.float32)

        # Compute velocity (moving length) during observation period
        moving_vector = _obs[..., -1, :] - _obs[..., 0, :]
        moving_length = torch.norm(moving_vector, dim=-1)   # (batch)

        # Compute pixel positions on seg maps
        W = seg_map_paras[..., :2][..., None, :]
        b = seg_map_paras[..., 2:4][..., None, :]

        # Compute angles and distances
        self.map_pos_pixel = self.map_pos_pixel.to(W.device)
        map_pos = (self.map_pos_pixel - b) / W

        # Compute distances and angles of all pixels
        direction_vectors = map_pos - current_pos           # (batch, a*a, 2)
        distances = torch.norm(direction_vectors, dim=-1)   # (batch, a*a)

        angles = torch.atan2(direction_vectors[..., 0],
                             direction_vectors[..., 1])     # (batch, a*a)
        angle_indices = (angles % (2*np.pi)) / (2*np.pi/self.partitions)
        angle_indices = angle_indices.to(torch.int32)

        # Compute the PhysicalCircle
        pc = []
        for r_times in self.radius:
            r = (r_times * moving_length)[..., None]
            radius_mask = (distances <= r).to(torch.float32)

            for ang in range(self.partitions):
                angle_mask = (angle_indices == ang).to(torch.float32)
                final_mask = radius_mask * angle_mask

                # Compute the minimum distance factor
                _d = (0 * map_safe_mask +
                      (1 - map_safe_mask) * final_mask * ((distances + MU) / (_maps + MU)))

                # Find the non-zero minimum value
                _zero_mask = (_d == 0).to(torch.float32)
                _d = (torch.ones_like(_d) * _zero_mask * INF +
                      _d * (1 - _zero_mask))
                _min_d, _ = torch.min(_d, dim=-1)

                _min_mask = (_min_d < INF).to(torch.float32)
                _min_d = _min_d * _min_mask
                pc.append(_min_d)

        # Final return shape: (batch, max_partitions, dim)
        pc = torch.stack(pc, dim=-1)
        pc = pc.reshape([-1, self.dim, self.partitions])
        pc = torch.transpose(pc, -2, -1)

        if (((m := self.max_partitions) is not None) and
                (m > (n := self.partitions))):
            paddings = [0, 0, 0, m - n, 0, 0]
            pc = torch.nn.functional.pad(pc, paddings)

        return pc

    def rotate(self, circle: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Rotate the physicalCircle. (Usually used after preprocess operations.)
        """
        # Rotate the circle <=> left or right shift the circle
        # Compute shift length
        angles = angles % (2*np.pi)
        partition_angle = (2*np.pi) / (self.partitions)
        move_length = (angles // partition_angle).to(torch.int32)

        # Remove paddings
        valid_circle = circle[..., :self.partitions, :]
        valid_circle = torch.concat([valid_circle, valid_circle], dim=-2)
        paddings = circle[..., self.partitions:, :]

        # Shift each circle
        rotated_circles = []
        for _circle, _move in zip(valid_circle, move_length):
            rotated_circles.append(_circle[_move:self.partitions+_move])

        rotated_circles = torch.stack(rotated_circles, dim=0)
        return torch.concat([rotated_circles, paddings], dim=-2)
