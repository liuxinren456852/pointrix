import torch

from dataclasses import dataclass
from pointrix.utils.base import BaseObject
from pointrix.utils.config import C
from .gaussian_utils import (
    inverse_sigmoid,
    build_rotation,
)

class DensificationContraller(BaseObject):
    @dataclass
    class Config:
        # Densification
        percent_dense: float = 0.01
        split_num: int = 2
        densify_stop_iter: int = 15000
        densify_start_iter: int = 500
        prune_interval: int = 100
        duplicate_interval: int = 100
        opacity_reset_interval: int = 3000
        densify_grad_threshold: float = 0.0002
        min_opacity: float = 0.005

    cfg: Config
    
    def setup(self, optimizer, point_cloud, cameras_extent) -> None:
        self.optimizer = optimizer
        self.point_cloud = point_cloud
        self.cameras_extent = cameras_extent
        
        # Densification setup
        num_points = len(self.point_cloud)
        self.max_radii2D = torch.zeros(num_points).to(self.device)
        self.percent_dense = self.cfg.percent_dense
        self.pos_gradient_accum = torch.zeros(
            (num_points, 1)
        ).to(self.device)
        self.denom = torch.zeros((num_points, 1)).to(self.device)
        self.opacity_deferred = False
        
    def updata_hypers(self, step):

        self.split_num = int(C(self.cfg.split_num, 0, step))
        self.prune_interval = int(C(self.cfg.prune_interval, 0, step))
        self.duplicate_interval = int(C(self.cfg.duplicate_interval, 0, step))
        self.opacity_reset_interval = int(C(self.cfg.opacity_reset_interval, 0, step))
        self.densify_grad_threshold = C(self.cfg.densify_grad_threshold, 0, step)
        self.min_opacity = C(self.cfg.min_opacity, 0, step)
        
    def update_state(self, step, visibility, viewspace_grad, radii, white_bg=False):
        if step < self.cfg.densify_stop_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility] = torch.max(
                self.max_radii2D[visibility],
                radii[visibility]
            )
            self.pos_gradient_accum[visibility] += torch.norm(
                viewspace_grad[visibility, :2],
                dim=-1,
                keepdim=True
            )
            self.denom[visibility] += 1
            
            if step > self.cfg.densify_start_iter:
                self.densification(step)
                
            # Reset opacity with a delay incase of validation right after reset opacity
            if self.opacity_deferred:
                self.opacity_deferred = False
                self.reset_opacity()

            if step % self.opacity_reset_interval == 0 or (white_bg and step == self.cfg.densify_start_iter):
                self.opacity_deferred = True
                
    def densification(self, step):
        if step % self.duplicate_interval == 0:
            grads = self.pos_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            self.densify_clone(grads)
            self.densify_split(grads)
        if step % self.prune_interval == 0:
            self.prune(step)
        torch.cuda.empty_cache()
        
    def reset_opacity(self):
        opc = self.point_cloud.get_opacity
        opacities_new = inverse_sigmoid(
            torch.min(opc, torch.ones_like(opc)*0.1)
        )
        self.point_cloud.replace(
            {"opacity": opacities_new},
            self.optimizer
        )

    def generate_clone_mask(self, grads):
        scaling = self.point_cloud.get_scaling
        cameras_extent = self.cameras_extent
        max_grad = self.densify_grad_threshold
        
        mask = torch.where(torch.norm(
            grads, dim=-1) >= max_grad, True, False)
        mask = torch.logical_and(
            mask,
            torch.max(
                scaling, 
                dim=1
            ).values <= self.percent_dense*cameras_extent
        )
        return mask
    
    def generate_split_mask(self, grads):
        scaling = self.point_cloud.get_scaling
        cameras_extent = self.cameras_extent
        max_grad = self.densify_grad_threshold
        
        num_points = len(self.point_cloud)
        padded_grad = torch.zeros((num_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        mask = torch.where(padded_grad >= max_grad, True, False)
        
        mask = torch.logical_and(
            mask,
            torch.max(
                scaling, 
                dim=1
            ).values > self.percent_dense*cameras_extent
        )
        return mask
        
    def new_pos_scale(self, mask):
        scaling = self.point_cloud.get_scaling
        position = self.point_cloud.get_position
        rotation = self.point_cloud.rotation
        split_num = self.split_num
        
        stds = scaling[mask].repeat(split_num, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            rotation[mask]
        ).repeat(split_num, 1, 1)
        new_pos = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        ) + (
            position[mask].repeat(split_num, 1)
        )
        new_scaling = self.point_cloud.scaling_inverse_activation(
            scaling[mask].repeat(split_num, 1) / (0.8*split_num)
        )
        return new_pos, new_scaling
    
    def densify_clone(self, grads):
        mask = self.generate_clone_mask(grads)
        atributes = self.point_cloud.select_atributes(mask)
        self.point_cloud.extand_points(atributes, self.optimizer)
        self.reset_densification_state()
        
    def densify_split(self, grads):
        mask = self.generate_split_mask(grads)
        new_pos, new_scaling = self.new_pos_scale(mask)
        atributes = self.point_cloud.select_atributes(mask)
        
        # Replace position and scaling from selected atributes
        atributes["position"] = new_pos
        atributes["scaling"] = new_scaling

        # Update rest of atributes
        for key, value in atributes.items():
            # Skip position and scaling, since they are already updated
            if key == "position" or key == "scaling":
                continue
            # Create a tuple of n_dim ones
            sizes = [1 for _ in range(len(value.shape))]
            sizes[0] = self.split_num
            sizes = tuple(sizes)

            # Repeat selected atributes in the fist dimension
            atributes[key] = value.repeat(*sizes)

        self.point_cloud.extand_points(atributes, self.optimizer)
        self.reset_densification_state()
        

    def prune(self, step):
        # TODO: fix me
        size_threshold = 20 if step > self.opacity_reset_interval else None
        cameras_extent = self.cameras_extent

        prune_filter = (
            self.point_cloud.get_opacity < self.min_opacity
        ).squeeze()
        if size_threshold:
            big_points_vs = self.max_radii2D > size_threshold
            big_points_ws = self.point_cloud.get_scaling.max(
                dim=1).values > 0.1 * cameras_extent
            prune_filter = torch.logical_or(prune_filter, big_points_vs)
            prune_filter = torch.logical_or(prune_filter, big_points_ws)

        valid_points_mask = ~prune_filter
        self.point_cloud.remove_points(valid_points_mask, self.optimizer)
        self.prune_postprocess(valid_points_mask)
        
    def prune_postprocess(self, valid_points_mask):
        self.pos_gradient_accum = self.pos_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
    def reset_densification_state(self):
        num_points = len(self.point_cloud)
        self.pos_gradient_accum = torch.zeros(
            (num_points, 1), device=self.device)
        self.denom = torch.zeros((num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((num_points), device=self.device)