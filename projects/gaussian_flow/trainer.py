import os
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from torch import nn

from pointrix.utils.losses import l1_loss, l2_loss, ssim
from pointrix.model.gaussian_splatting import GaussianSplatting
from pointrix.model.gaussian_utils import (
    validation_process,
    render_batch,
)

from .gf_densification import FlowDensificationContraller

class GaussianFlow(GaussianSplatting):
    @dataclass
    class Config(GaussianSplatting.Config):
        pass
    
    cfg: Config
    
    def setup(self, cfg):
        super().setup(cfg)
        
        # Set up flow paramters
        self.max_timestamp = self.datapipline.training_dataset.max_timestamp
        self.point_cloud.max_timestamp = self.max_timestamp
        self.point_cloud.max_steps = self.cfg.max_steps
        
    def before_train_start(self):
        # Densification setup
        self.densification_control = FlowDensificationContraller(
            self.cfg.densification,
            optimizer=self.optimizer,
            point_cloud=self.point_cloud,
            cameras_extent=self.cameras_extent,
        )
        self.densification_control.updata_hypers(step=0)
        
        self.point_cloud = self.point_cloud.to(self.device)
        
    def train_step(self, batch) -> Dict:
        self.update_sh_degree()
        atributes_dict = {
            "position": self.point_cloud.get_position_flow,
            "rotation": self.point_cloud.get_rotation_flow,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "shs": self.point_cloud.get_shs,
            "active_sh_degree": self.active_sh_degree,
            "bg_color": self.background,
        }

        def render_func(data):
            data.update(atributes_dict)
            self.point_cloud.set_timestamp(
                t=data["camera"].timestamp,
                training=True,
                training_steps=self.global_step,
            )
            return self.renderer(**data)
        
        (
            images,
            self.radii,
            self.visibility,
            viewspace_points
        ) = render_batch(render_func, batch)
        gt_images = torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))], 
            dim=0
        )        
        
        L1_loss = l1_loss(images, gt_images)
        # ssim_loss = 1.0 - ssim(images, gt_images)
        loss = (
            (1.0 - self.cfg.lambda_dssim) * L1_loss
        ) + (
            # self.cfg.lambda_dssim * ssim_loss
        )

        loss.backward()

        self.accumulate_viewspace_grad(viewspace_points)

        # print("viewspace_grad: ", self.viewspace_grad)
        # TODO: log the learning rate of each elements in optimizer
        for param_group in self.optimizer.param_groups:
            name = param_group['name']
            if name == "point_cloud.position":
                pos_lr = param_group['lr']
                break

        return {
            "loss": loss,
            "L1_loss": L1_loss,
            # "ssim_loss": ssim_loss,
            "num_pt": len(self.point_cloud),
            "pos_lr": pos_lr,
        }
    
    @torch.no_grad()
    def val_step(self):
        
        atributes_dict = {
            "position": self.point_cloud.get_position_flow,
            "rotation": self.point_cloud.get_rotation_flow,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "shs": self.point_cloud.get_shs,
            "active_sh_degree": self.active_sh_degree,
            "bg_color": self.background,
        }

        def render_func(data):
            data.update(atributes_dict)
            self.point_cloud.set_timestamp(
                t=data["camera"].timestamp,
                training=True,
                training_steps=self.global_step,
            )
            return self.renderer(**data)

        validation_process(
            render_func,
            self.datapipline,
            self.global_step,
            self.logger
        )