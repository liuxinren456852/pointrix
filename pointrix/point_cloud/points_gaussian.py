import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from torch import nn
from pointrix.utils.sh_utils import RGB2SH
from pointrix.engine.trainer import DefaultTrainer
from pointrix.point_cloud.points import PointsCloud
from pointrix.point_cloud.base import BaseObject
from pointrix.utils.losses import l1_loss, l2_loss, ssim
from pointrix.utils.optimizer import parse_scheduler, parse_optimizer
from pointrix.point_cloud.utils import (
        build_covariance_from_scaling_rotation,
        inverse_sigmoid,
        build_rotation,
)

from simple_knn._C import distCUDA2


def gaussian_point_init(position, max_sh_degree, device):
    num_points = len(position)
    dist2 = torch.clamp_min(distCUDA2(position), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((num_points, 4)).to(device)
    rots[:, 0] = 1

    init_one = torch.ones(
        (num_points, 1),
        dtype=torch.float32
    ).to(device)
    opacities = inverse_sigmoid(0.1 * init_one)
    features_rest = torch.zeros(
        (num_points, (max_sh_degree+1) ** 2 - 1, 3),
        dtype=torch.float32
    ).to(device)

    return scales, rots, opacities, features_rest


class GaussianSplatting(BaseObject):
    @dataclass
    class Config:
        max_sh_degree: int = 3
        percent_dense: float = 0.01
        densify_grad_threshold: float = 0.0002
        min_opacity: float = 0.005
        optimizer: dict = field(default_factory=dict)
        scheduler: dict = field(default_factory=dict)
        points_cloud: dict = field(default_factory=dict)

    cfg: Config

    def saving(self):
        data_list = {
            "active_sh_degree": self.active_sh_degree,
            "points_cloud": self.points_cloud.state_dict(),
        }
        return data_list

    def setup(self, point_cloud, radius):
        # Activation funcitons
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # Training variables
        self.active_sh_degree = 0
        self.size_threshold = None

        # Set up scheduler for points cloud position
        self.cameras_extent = radius
        if self.cfg.scheduler is not None:
            self.schedulers = parse_scheduler(
                self.cfg.scheduler, self.cameras_extent)

        # Set up points cloud
        self.points_cloud = PointsCloud(
            self.cfg.points_cloud, point_cloud).to(self.device)
        num_points = len(self.points_cloud)

        scales, rots, opacities, features_rest = gaussian_point_init(
            self.points_cloud.position,
            self.cfg.max_sh_degree,
            self.device
        )

        fused_color = self.points_cloud.features.unsqueeze(1)
        self.points_cloud.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )

        self.points_cloud.register_atribute("features_rest", features_rest)
        self.points_cloud.register_atribute("scaling", scales)
        self.points_cloud.register_atribute("rotation", rots)
        self.points_cloud.register_atribute("opacity", opacities)

        # Densification setup
        self.max_radii2D = torch.zeros(num_points).to(self.device)
        self.percent_dense = self.cfg.percent_dense
        self.pos_gradient_accum = torch.zeros(
            (num_points, 1)
        ).to(self.device)
        self.denom = torch.zeros((num_points, 1)).to(self.device)

        self.optimizer = parse_optimizer(self.cfg.optimizer, self.points_cloud)

    def update_sh_degree(self):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.active_sh_degree < self.cfg.max_sh_degree:
            self.active_sh_degree += 1

    @torch.no_grad()
    def accumulate_viewspace_grad(self, viewspace_points):
        # Accumulate viewspace gradients for batch
        self.viewspace_grad = torch.zeros_like(
            viewspace_points[0]
        )
        for vp in viewspace_points:
            self.viewspace_grad += vp.grad

    # TODO: make this function more readable and modular
    def densify(self):

        grads = self.pos_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        cameras_extent = self.cameras_extent
        max_grad = self.cfg.densify_grad_threshold
        num_points = len(self.points_cloud)
        split_num = 2

        # densify and clone
        selected_pts_mask = torch.where(torch.norm(
            grads, dim=-1) >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling,
                      dim=1).values <= self.percent_dense*cameras_extent
        )

        select_atributes = self.points_cloud.select_atributes(
            selected_pts_mask)
        self.points_cloud.extand_points(select_atributes, self.optimizer)
        self.reset_densification_state()

        # densify and split
        # Extract points that satisfy the gradient condition
        num_points = len(self.points_cloud)
        padded_grad = torch.zeros((num_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= max_grad, True, False)

        position = self.get_position
        scaling = self.get_scaling
        rotation = self.points_cloud.rotation
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scaling, dim=1).values > self.percent_dense *
            cameras_extent
        )
        stds = scaling[selected_pts_mask].repeat(split_num, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            rotation[selected_pts_mask]
        ).repeat(split_num, 1, 1)
        new_pos = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        ) + (
            position[selected_pts_mask].repeat(split_num, 1)
        )
        new_scaling = self.scaling_inverse_activation(
            scaling[selected_pts_mask].repeat(split_num, 1) / (0.8*split_num)
        )

        select_atributes = self.points_cloud.select_atributes(
            selected_pts_mask)

        # Replace position and scaling from selected atributes
        select_atributes["position"] = new_pos
        select_atributes["scaling"] = new_scaling

        # Update rest of atributes
        for key, value in select_atributes.items():
            # Skip position and scaling, since they are already updated
            if key == "position" or key == "scaling":
                continue
            # Create a tuple of n_dim ones
            sizes = [1 for _ in range(len(value.shape))]
            sizes[0] = split_num
            sizes = tuple(sizes)

            # Repeat selected atributes in the fist dimension
            select_atributes[key] = value.repeat(*sizes)

        self.points_cloud.extand_points(select_atributes, self.optimizer)
        self.reset_densification_state()

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(
                split_num * selected_pts_mask.sum(),
                device=self.device,
                dtype=bool
            )
        ))
        valid_points_mask = ~prune_filter
        self.points_cloud.remove_points(valid_points_mask, self.optimizer)
        self.prune_postprocess(valid_points_mask)

    def prune(self):
        # TODO: fix me
        cameras_extent = self.cameras_extent

        prune_filter = (
            self.get_opacity < self.cfg.min_opacity
        ).squeeze()
        if self.size_threshold:
            big_points_vs = self.max_radii2D > self.size_threshold
            big_points_ws = self.get_scaling.max(
                dim=1).values > 0.1 * cameras_extent
            prune_filter = torch.logical_or(prune_filter, big_points_vs)
            prune_filter = torch.logical_or(prune_filter, big_points_ws)

        valid_points_mask = ~prune_filter
        self.points_cloud.remove_points(valid_points_mask, self.optimizer)
        self.prune_postprocess(valid_points_mask)

    def reset_densification_state(self):
        num_points = len(self.points_cloud)
        self.pos_gradient_accum = torch.zeros(
            (num_points, 1), device=self.device)
        self.denom = torch.zeros((num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((num_points), device=self.device)

    def prune_postprocess(self, valid_points_mask):
        self.pos_gradient_accum = self.pos_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def reset_opacity(self):
        opc = self.get_opacity
        opacities_new = inverse_sigmoid(
            torch.min(opc, torch.ones_like(opc)*0.1)
        )
        self.points_cloud.replace(
            {"opacity": opacities_new},
            self.optimizer
        )

    @property
    def get_opacity(self):
        return self.opacity_activation(self.points_cloud.opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self.points_cloud.scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.points_cloud.rotation)

    @property
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling,
            scaling_modifier,
            self.points_cloud.rotation,
        )

    @property
    def get_shs(self):
        return torch.cat([
            self.points_cloud.features,
            self.points_cloud.features_rest,
        ], dim=1)

    @property
    def get_position(self):
        return self.points_cloud.position
