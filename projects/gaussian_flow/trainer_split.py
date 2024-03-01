import os
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from torch import nn

from pointrix.point_cloud import parse_point_cloud
from pointrix.utils.losses import l1_loss, l2_loss, ssim
from pointrix.model.gaussian_densification import DensificationContraller
from pointrix.model.gaussian_splatting import GaussianSplatting
from pointrix.model.gaussian_utils import (
    validation_process,
    render_batch,
)

from workspace.code.Pointrix.projects.gaussian_flow.gf_point_optimizer import FlowDensificationContraller

class GaussianFlow(GaussianSplatting):
    @dataclass
    class Config(GaussianSplatting.Config):
        point_cloud_flow: dict = field(default_factory=dict)
    
    cfg: Config
    
    def setup(self):
        super().setup()
        
        # Set up flow paramters
        self.max_timestamp = self.datapipline.training_dataset.max_timestamp
        self.point_cloud.max_timestamp = self.max_timestamp
        self.point_cloud.max_steps = self.cfg.max_steps
        
        self.point_cloud_flow = parse_point_cloud(
            self.cfg.point_cloud_flow, 
            self.datapipline
        )
        
    def before_train_start(self):
        # Densification setup
        self.flow_densification_control = FlowDensificationContraller(
            self.cfg.densification,
            optimizer=self.optimizer,
            point_cloud=self.point_cloud_flow,
            cameras_extent=self.cameras_extent,
        )
        self.flow_densification_control.updata_hypers(step=0)
        
        self.densification_control = DensificationContraller(
            self.cfg.densification,
            optimizer=self.optimizer,
            point_cloud=self.point_cloud,
            cameras_extent=self.cameras_extent,
        )
        self.densification_control.updata_hypers(step=0)
        
        self.point_cloud = self.point_cloud.to(self.device)
        self.point_cloud_flow = self.point_cloud_flow.to(self.device)
        
    def fuse_gaussian(self):
        atributes_dict = {
            "opacity": torch.cat([
                self.point_cloud.get_opacity,
                self.point_cloud_flow.get_opacity,
            ], dim=0),
            "scaling": torch.cat([
                self.point_cloud.get_scaling,
                self.point_cloud_flow.get_scaling,
            ], dim=0),
            "shs": torch.cat([
                self.point_cloud.get_shs,
                self.point_cloud_flow.get_shs,
            ], dim=0),
            "active_sh_degree": self.active_sh_degree,
            "bg_color": self.background,
        }
        return atributes_dict
    
    def fuse_flow(self):
        atributes_dict = {
            "position": torch.cat([
                self.point_cloud.get_position,
                self.point_cloud_flow.get_position_flow,
            ], dim=0),
            "rotation": torch.cat([
                self.point_cloud.get_rotation,
                self.point_cloud_flow.get_rotation_flow,
            ], dim=0),
        }
        return atributes_dict
        
    def train_step(self, batch) -> Dict:
        self.update_sh_degree()
        atributes_dict = self.fuse_gaussian()

        def render_func(data):
            data.update(atributes_dict)
            self.point_cloud_flow.set_timestep(
                t=data["camera"].timestamp,
                training=True,
                training_step=self.global_step,
            )
            timestamp_dict = self.fuse_flow()
            data.update(timestamp_dict)
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
        # loss = (
        #     (1.0 - self.cfg.lambda_dssim) * L1_loss
        # ) + (
        #     self.cfg.lambda_dssim * ssim_loss
        # )
        loss = L1_loss

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
            "num_pt": len(self.point_cloud) + len(self.point_cloud_flow),
            "pos_lr": pos_lr,
        }
    
    @torch.no_grad()
    def val_step(self):
        
        atributes_dict = self.fuse_gaussian()

        def render_func(data):
            data.update(atributes_dict)
            self.point_cloud_flow.set_timestep(
                t=data["camera"].timestamp,
            )
            timestamp_dict = self.fuse_flow()
            data.update(timestamp_dict)
            return self.renderer(**data)

        validation_process(
            render_func,
            self.datapipline,
            self.global_step,
            self.logger
        )
        
    def upd_bar_info(self, info: dict) -> None:
        pt = len(self.point_cloud) + len(self.point_cloud_flow)
        info.update({
            "num_pt": f"{pt}",
        })
        
    @torch.no_grad()
    def update_state(self) -> None:   
        pt = len(self.point_cloud)
        pt_flow = len(self.point_cloud_flow) 
        self.densification_control.update_state(
            step=self.global_step,
            visibility=self.visibility[:pt],
            viewspace_grad=self.viewspace_grad[:pt],
            radii=self.radii[:pt],
            white_bg=self.white_bg,
        )
        self.densification_control.updata_hypers(
            step=self.global_step,
        )
        
        self.flow_densification_control.update_state(
            step=self.global_step,
            visibility=self.visibility[pt:],
            viewspace_grad=self.viewspace_grad[pt:],
            radii=self.radii[pt:],
            white_bg=self.white_bg,
        )
        self.flow_densification_control.updata_hypers(
            step=self.global_step,
        )