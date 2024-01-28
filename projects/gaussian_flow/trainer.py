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
    video_process,
    render_batch,
)

from gf_densification import FlowDensificationContraller

class GaussianFlow(GaussianSplatting):
    @dataclass
    class Config(GaussianSplatting.Config):
        lambda_param_l1: float = 0.0
        lambda_knn: float = 0.0
        grad_clip_value: float = 0.0
    
    cfg: Config
    
    def setup(self):
        super().setup()
        
        # Set up flow paramters
        self.max_timestamp = self.datapipline.training_dataset.max_timestamp
        self.point_cloud.max_timestamp = self.max_timestamp
        self.point_cloud.max_steps = self.cfg.max_steps
        
    def update_sh_degree(self):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.global_step % 100 == 0:
            if self.active_sh_degree < self.cfg.max_sh_degree:
                self.active_sh_degree += 1
        
    def before_train_start(self):
        # Densification setup
        self.densification_control = FlowDensificationContraller(
            self.cfg.densification,
            optimizer=self.optimizer,
            point_cloud=self.point_cloud,
            cameras_extent=self.cameras_extent,
        )
        self.densification_control.updata_hypers(step=0)
        self.after_densifi_step = self.cfg.densification.densify_stop_iter+1
        
        self.point_cloud = self.point_cloud.to(self.device)
        
    def get_gaussian(self):
        atributes_dict = {
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "active_sh_degree": self.active_sh_degree,
            "bg_color": self.background,
        }
        return atributes_dict
    
    def get_flow(self):
        atributes_dict = {
            "position": self.point_cloud.get_position_flow,
            "rotation": self.point_cloud.get_rotation_flow,
            "shs": self.point_cloud.get_shs_flow,
        }
        return atributes_dict
    
    def params_l1_regulizer(self):
        # random_choice = torch.sample (
        #     0, len(self.point_cloud), (10000, )
        # )
        # pos = self.point_cloud.position[:, 3:]
        # rot = self.point_cloud.rotation[:, 4:]
        pos = self.point_cloud.pos_params
        rot = self.point_cloud.rot_params
        pos_abs = torch.abs(pos)
        # pos_norm = pos_abs / pos_abs.max(dim=1, keepdim=True)[0]
        
        rot_abs = torch.abs(rot)
        # rot_norm = rot_abs / rot_abs.max(dim=1, keepdim=True)[0]
        
        loss_l1 = pos_abs.mean() + rot_abs.mean()
        # loss_norm = pos_norm.mean() + rot_norm.mean()
        
        return loss_l1 
        
    def train_step(self, batch) -> Dict:
        self.update_sh_degree()
        
        atributes_dict = self.get_gaussian()
        
        def render_func(data):
            data.update(atributes_dict)
            self.point_cloud.set_timestep(
                t=data["camera"].timestamp,
                training=True,
                training_step=self.global_step,
            )
            timestamp_dict = self.get_flow()
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
        result_dict = {}
        loss = 0.
        L1_loss = l1_loss(images, gt_images)
        result_dict.update({
            "L1_loss": L1_loss,
            "num_pt": len(self.point_cloud),
        })
        loss += L1_loss
        
        if self.cfg.lambda_param_l1 > 0:
            param_l1 = self.params_l1_regulizer()
            result_dict.update({
                "pl1_loss": param_l1,
            })
            loss += self.cfg.lambda_param_l1 * param_l1
            
        if self.cfg.lambda_knn > 0:
            if self.global_step == self.after_densifi_step:
                self.point_cloud.gen_knn()
                
            if self.global_step > self.after_densifi_step:
                knn_loss = self.point_cloud.knn_loss()
                result_dict.update({
                    "knn_loss": knn_loss,
                })
                loss += self.cfg.lambda_knn * knn_loss
                
        # ssim_loss = 1.0 - ssim(images, gt_images)
        # result_dict.update({
        #     "ssim_loss": ssim_loss,
        # })
        # loss = (
        #     (1.0 - self.cfg.lambda_dssim) * L1_loss
        # ) + (
        #     self.cfg.lambda_dssim * ssim_loss
        # )
        loss.backward()
        
        if self.cfg.grad_clip_value > 0:
            torch.nn.utils.clip_grad_value_(
                self.point_cloud.parameters(), 
                self.cfg.grad_clip_value
            )

        self.accumulate_viewspace_grad(viewspace_points)

        # print("viewspace_grad: ", self.viewspace_grad)
        # TODO: log the learning rate of each elements in optimizer
        for param_group in self.optimizer.param_groups:
            name = param_group['name']
            if name == "point_cloud.position":
                pos_lr = param_group['lr']
                result_dict.update({"pos_lr": pos_lr,})
                break
        
        result_dict.update({"loss": loss, })
        return result_dict
    
    @torch.no_grad()
    def val_step(self):
        
        atributes_dict = self.get_gaussian()

        def render_func(data):
            data.update(atributes_dict)
            self.point_cloud.set_timestep(
                t=data["camera"].timestamp,
            )
            timestamp_dict = self.get_flow()
            data.update(timestamp_dict)
            return self.renderer(**data)

        validation_process(
            render_func,
            self.datapipline,
            self.global_step,
            self.logger
        )
        
    @torch.no_grad()
    def video_step(self, video_path, save_npz=False):
        atributes_dict = self.get_gaussian()

        def render_func(data):
            data.update(atributes_dict)
            self.point_cloud.set_timestep(
                t=data["camera"].timestamp,
            )
            timestamp_dict = self.get_flow()
            data.update(timestamp_dict)
            return self.renderer(**data)

        video_process(
            render_func,
            self.datapipline,
            video_path,
            save_npz=save_npz,
            pcd=self.point_cloud,
        )