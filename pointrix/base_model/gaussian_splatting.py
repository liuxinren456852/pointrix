import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from torch import nn
from pointrix.utils.sh_utils import RGB2SH
from pointrix.engine.trainer import DefaultTrainer
from pointrix.points import PointsCloud
from pointrix.utils.losses import l1_loss, l2_loss, ssim
from pointrix.utils.optimizer import parse_scheduler
from pointrix.base_model.gaussian_utils import (
    build_covariance_from_scaling_rotation,
    inverse_sigmoid,
    build_rotation,
    validation_process,
    render_batch,
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


class GaussianSplatting(DefaultTrainer):
    @dataclass
    class Config(DefaultTrainer.Config):
        max_sh_degree: int = 3

        # Train cfg
        percent_dense: float = 0.01
        lambda_dssim: float = 0.2
        spatial_lr_scale: bool = True

        # Densification
        densify_stop_iter: int = 15000
        densify_start_iter: int = 500
        prune_interval: int = 100
        duplicate_interval: int = 100
        opacity_reset_interval: int = 3000
        densify_grad_threshold: float = 0.0002
        min_opacity: float = 0.005

    cfg: Config

    def setup(self, point_cloud):
        # Activation funcitons
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # Training variables
        self.active_sh_degree = 0

        # Set up scheduler for points cloud position
        self.cameras_extent = self.datapipline.training_dataset.radius
        if self.cfg.scheduler is not None:
            self.schedulers = parse_scheduler(
                self.cfg.scheduler, self.cameras_extent)

        # Set up points cloud
        self.points_cloud = PointsCloud(self.cfg.points_cloud, point_cloud).to(self.device)
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

        # TODO: use camera to get the extent
        self.white_bg = True

        bg_color = [1, 1, 1] if self.white_bg else [0, 0, 0]
        self.background = torch.tensor(
            bg_color,
            dtype=torch.float32,
            device=self.device,
        )

    def train_step(self, batch) -> Dict:
        self.update_sh_degree()

        def render_func(data):
            render_pkg = self.renderer(
                **data,
                position=self.get_position,
                opacity=self.get_opacity,
                scaling=self.get_scaling,
                rotation=self.get_rotation,
                shs=self.get_shs,
                active_sh_degree=self.active_sh_degree,
                bg_color=self.background,
            )
            return render_pkg
        
        (
            images,
            self.radii,
            self.visibility,
            viewspace_points
        ) = render_batch(render_func, batch)
        gt_images = torch.cat([batch[i]["image"].cuda().unsqueeze(0) for i in range(len(batch))], dim=0).float()
        L1_loss = l1_loss(images, gt_images)
        ssim_loss = 1.0 - ssim(images, gt_images)
        loss = (
            (1.0 - self.cfg.lambda_dssim) * L1_loss
        ) + (
            self.cfg.lambda_dssim * ssim_loss
        )

        loss.backward()

        self.accumulate_viewspace_grad(viewspace_points)

        # print("viewspace_grad: ", self.viewspace_grad)
        # TODO: log the learning rate of each elements in optimizer
        for param_group in self.optimizer.param_groups:
            name = param_group['name']
            if name == "points_cloud.position":
                pos_lr = param_group['lr']
                break

        return {
            "loss": loss,
            "L1_loss": L1_loss,
            "ssim_loss": ssim_loss,
            "num_pt": len(self.points_cloud),
            "pos_lr": pos_lr,
        }

    def update_sh_degree(self):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.global_step % 1000 == 0:
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

    @torch.no_grad()
    def val_step(self):
        def render_func(data):
            render_pkg = self.renderer(
                **data,
                position=self.get_position,
                opacity=self.get_opacity,
                scaling=self.get_scaling,
                rotation=self.get_rotation,
                shs=self.get_shs,
                active_sh_degree=self.active_sh_degree,
                bg_color=self.background,
            )
            return render_pkg

        validation_process(
            render_func,
            self.datapipline,
            self.global_step,
            self.logger
        )

    @torch.no_grad()
    def test(self):
        pass

    @torch.no_grad()
    def update_state(self) -> None:
        if self.global_step < self.cfg.densify_stop_iter:
            # Keep track of max radii in image-space for pruning
            visibility = self.visibility
            self.max_radii2D[visibility] = torch.max(
                self.max_radii2D[visibility],
                self.radii[visibility]
            )
            self.pos_gradient_accum[visibility] += torch.norm(
                self.viewspace_grad[visibility, :2],
                dim=-1,
                keepdim=True
            )
            self.denom[visibility] += 1
            if self.global_step > self.cfg.densify_start_iter:
                # import pdb; pdb.set_trace()
                self.densification()

            if self.global_step % self.cfg.opacity_reset_interval == 0 or (self.white_bg and self.global_step == self.cfg.densify_start_iter):
                self.reset_opacity()
        super().update_state()

    def densification(self):
        if self.global_step % self.cfg.duplicate_interval == 0:
            self.densify()
        if self.global_step % self.cfg.prune_interval == 0:
            self.prune()
        torch.cuda.empty_cache()

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
        size_threshold = 20 if self.global_step > self.cfg.opacity_reset_interval else None
        cameras_extent = self.cameras_extent

        prune_filter = (
            self.get_opacity < self.cfg.min_opacity
        ).squeeze()
        if size_threshold:
            big_points_vs = self.max_radii2D > size_threshold
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
            self.get_rotation,
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
