import os
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from torch import nn
from pointrix.engine.trainer import DefaultTrainer
from pointrix.points import PointsCloud
from pointrix.utils.losses import l1_loss, l2_loss, ssim
from pointrix.base_model.gaussian_utils import (
    build_covariance_from_scaling_rotation, 
    inverse_sigmoid,
    build_rotation,
)

from simple_knn._C import distCUDA2

def gaussian_point_init(position, max_sh_degree, device):
    num_points = len(position)
    dist2 = torch.clamp_min(distCUDA2(position), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    rots = torch.zeros((num_points, 4)).to(device)
    rots[:, 0] = 1

    init_one = torch.ones(
        (num_points, 1), 
        dtype=torch.float32
    ).to(device)
    opacities = inverse_sigmoid(0.1 * init_one)
    features_rest = torch.zeros(
        (num_points, max_sh_degree ** 2, 3), 
        dtype=torch.float32
    ).to(device)
    
    return scales, rots, opacities, features_rest

class GaussianSplatting(DefaultTrainer):
    @dataclass
    class Config:
        max_sh_degree: int
        
        # Train cfg
        percent_dense: float
        lambda_dssim: float
        
        # Densification
        densify_stop_iter: int
        densify_start_iter: int
        prune_interval: int
        duplicate_interval: int
        opacity_reset_interval: int
        densify_grad_threshold: float
        min_opacity: float
    
    cfg: Config
    
    def setup(self):
        # Activation funcitons
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
        # Training variables
        self.active_sh_degree = 0
        
        # Set up points cloud
        self.points_cloud = PointsCloud(self.cfg.points_cloud).to(self.cfg.device)
        num_points = len(self.points_cloud)
        
        scales, rots, opacities, features_rest = gaussian_point_init(
            self.points_cloud.xyz, 
            self.cfg.max_sh_degree, 
            self.cfg.device
        )
        
        self.points_cloud.register_atribute("features_rest", features_rest)
        self.points_cloud.register_atribute("scaling", scales)
        self.points_cloud.register_atribute("rotation", rots)
        self.points_cloud.register_atribute("opacity", opacities)    
        
        # Densification setup
        self.max_radii2D = torch.zeros(num_points, ).to(self.cfg.device)
        self.percent_dense = self.cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1)).to(self.cfg.device)
        self.denom = torch.zeros((self._xyz.shape[0], 1)).to(self.cfg.device)
        self.cameras_extent = self.dataloader['val'].cameras_extent
    
    def train_step(self, batch) -> Dict:
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.global_step % 1000 == 0:
            self.active_sh_degree += 1
        
        batch_size = len(batch)
        bg_color = [1, 1, 1] if self.cfg.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        renders = []
        viewspace_points = []
        visibilitys = []
        radiis = []
        gt_images = []
        for b_i in batch:
            # Get all gaussian parameters
            position = self.points_cloud.position
            opacity = self.opacity_activation(
                self.points_cloud.opacity
            )
            scaling = self.scaling_activation(
                self.points_cloud.scaling
            )
            rotation = self.rotation_activation(
                self.points_cloud.rotation
            )
            shs = torch.cat([
                self.points_cloud.features,
                self.points_cloud.features_rest,
            ], dim=-1)
            
            render_pkg = self.renderer(
                position = position,
                opacity = opacity,
                scaling = scaling,
                rotation = rotation,
                shs = shs,
                active_sh_degree=self.active_sh_degree,
                background=background,
                **b_i
            )
            
        renders.append(render_pkg["render"])
        viewspace_points.append(render_pkg["viewspace_points"])
        visibilitys.append(render_pkg["visibility_filter"].unsqueeze(0))
        radiis.append(render_pkg["radii"].unsqueeze(0))
        gt_images.append(b_i["image"].cuda())
        
        self.radii = torch.cat(radiis,0).max(dim=0).values
        self.visibility = torch.cat(visibilitys).any(dim=0)
        images = torch.stack(renders)
        gt_images = torch.stack(gt_images)    
        

        L1_loss = l1_loss(images, gt_images) 
        ssim_loss = ssim(images, gt_images)
        loss = (
            (1.0 - self.cfg.lambda_dssim) * L1_loss
        ) + (
            self.cfg.lambda_dssim * (1.0 - ssim_loss)
        )
        
        loss.backward()
        
        # Accumulate viewspace gradients for batch
        self.viewspace_grad = torch.zeros_like(
            viewspace_points[0]
        )
        for vp in viewspace_points:
            self.viewspace_grad += vp.grad
            
        return {
            "loss": loss,
            "L1_loss": L1_loss,
            "ssim_loss": ssim_loss,
        }
    
    def val_step(self):
        pass
    
    def test_step(self):
        pass
    
    def upd_bar_info(self, info: dict) -> None:
        info.update({
            "Num_pt": len(self.points_cloud),
        })
    
    @torch.no_grad()
    def update_state(self) -> None:
        super.update_state()
        
        if self.global_step < self.cfg.densify_stop_iter:
            # Keep track of max radii in image-space for pruning
            visibility = self.visibility
            radii = self.radii
            self.max_radii2D[visibility] = torch.max(
                self.max_radii2D[visibility], 
                radii[visibility]
            )
            self.xyz_gradient_accum[visibility] += torch.norm(
                self.viewspace_grad[visibility,:2], 
                dim=-1, 
                keepdim=True
            )
            self.denom[visibility] += 1
            if self.global_step > self.cfg.densify_start_iter:
                self.densification()
    
    def densification(self):
        if self.global_step % self.cfg.densification_interval == 0:
            self.densify()
        if self.global_step % self.cfg.prune_interval == 0:
            self.prune()
        torch.cuda.empty_cache()
            
    def densify(self):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        cameras_extent = self.cameras_extent
        max_grad = self.cfg.densify_grad_threshold
        num_points = len(self.points_cloud)
        split_num = 2
        
        # densify and clone
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*cameras_extent
        )
        
        select_atributes = self.points_cloud.select_atributes(selected_pts_mask)
        self.points_cloud.densify(select_atributes, self.optimizer)
        
        # densify and split
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((num_points), device=self.cfg.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= max_grad, True, False)
        
        position = self.points_cloud.position
        scaling = self.scaling_activation(
            self.points_cloud.scaling
        )
        rotation = self.points_cloud.rotation
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scaling, dim=1).values > self.percent_dense*cameras_extent
        )
        stds = scaling[selected_pts_mask].repeat(split_num, 1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            rotation[selected_pts_mask]
        ).repeat(split_num,1,1)
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        ) + (
            position[selected_pts_mask].repeat(split_num, 1)
        )
        new_scaling = self.scaling_inverse_activation(
            scaling[selected_pts_mask].repeat(split_num, 1) / (0.8*split_num)
        )
        
        select_atributes = self.points_cloud.select_atributes(selected_pts_mask)
        
        # Replace position and scaling from selected atributes
        select_atributes["position"] = new_xyz
        select_atributes["scaling"] = new_scaling
        
        # Update rest of atributes
        for key, value in select_atributes.items():
            # Skip position and scaling, since they are already updated
            if key in ["position", "scaling"]:
                continue
            # Create a tuple of n_dim ones
            sizes = [1 for n_d in range(len(value.shape))]
            sizes[0] = split_num
            sizes = tuple(sizes)
            
            # Repeat selected atributes in the fist dimension
            select_atributes[key] = value.repeat(*sizes)

        self.points_cloud.densify(select_atributes, self.optimizer)
        
        prune_filter = torch.cat((
            selected_pts_mask, 
            torch.zeros(
                split_num * selected_pts_mask.sum(), 
                device=self.cfg.device, 
                dtype=bool
            )
        ))
        valid_points_mask = ~prune_filter
        self.points_cloud.prune(valid_points_mask, self.optimizer)
        
        self.reset_densification_state()
        
    
    def prune(self):
        # TODO: fix me
        size_threshold = 20 if self.global_step > self.cfg.opacity_reset_interval else None
        cameras_extent = self.cameras_extent
        
        prune_filter = (
            self.get_opacity < self.cfg.min_opacity
        ).squeeze()
        if size_threshold:
            big_points_vs = self.max_radii2D > size_threshold
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * cameras_extent
            prune_filter = torch.logical_or(prune_filter, big_points_vs)
            prune_filter = torch.logical_or(prune_filter, big_points_ws)
        
        valid_points_mask = ~prune_filter
        self.points_cloud.prune(valid_points_mask, self.optimizer)
        self.pruned_postprocess(valid_points_mask)
        
    def reset_densification_state(self):
        num_points = len(self.points_cloud)
        self.xyz_gradient_accum = torch.zeros((num_points, 1), device=self.cfg.device)
        self.denom = torch.zeros((num_points, 1), device=self.cfg.device)
        self.max_radii2D = torch.zeros((num_points), device=self.cfg.device)
        
    def prune_postprocess(self, valid_points_mask):
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]