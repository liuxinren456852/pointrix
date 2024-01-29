from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from pointrix.utils.sh_utils import RGB2SH
from pointrix.engine.default_trainer import DefaultTrainer
from pointrix.utils.losses import l1_loss, l2_loss, ssim
from pointrix.exporter.novel_view import test_view_render, novel_view_render
from .gaussian_utils import (
    validation_process,
    render_batch,
)

class GaussianSplatting(DefaultTrainer):
    @dataclass
    class Config(DefaultTrainer.Config):
        max_sh_degree: int = 3
        # Train cfg
        lambda_dssim: float = 0.2        
        densification: dict = field(default_factory=dict)

    cfg: Config

    def train_step(self, batch) -> Dict:
        self.update_sh_degree()

        atributes_dict = {
            "position": self.point_cloud.position,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation,
            "shs": self.point_cloud.get_shs,
            "active_sh_degree": self.active_sh_degree,
            "bg_color": self.background,
        }

        def render_func(data):
            data.update(atributes_dict)
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
        ssim_loss = 1.0 - ssim(images, gt_images)
        loss = (
            (1.0 - self.cfg.lambda_dssim) * L1_loss
        ) + (
            self.cfg.lambda_dssim * ssim_loss
        )

        # print("viewspace_grad: ", self.viewspace_grad)
        # TODO: log the learning rate of each elements in optimizer
        for param_group in self.optimizer.param_groups:
            name = param_group['name']
            if name == "position":
                pos_lr = param_group['lr']
                break
        loss_dict = {"loss": loss,
                    "L1_loss": L1_loss,
                    "ssim_loss": ssim_loss,
                    "num_pt": len(self.point_cloud),
                    "pos_lr": pos_lr,},
        optimizer_dict = {"loss": loss,
                        "viewspace_points": viewspace_points,
                        "visibility": self.visibility,
                        "radii": self.radii,
                        "white_bg":self.white_bg}
        return loss_dict, optimizer_dict
                
    def update_sh_degree(self):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.global_step % 1000 == 0:
            if self.active_sh_degree < self.cfg.max_sh_degree:
                self.active_sh_degree += 1

    @torch.no_grad()
    def val_step(self):
        
        atributes_dict = {
            "position": self.point_cloud.position,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
            "rotation": self.point_cloud.get_rotation,
            "shs": self.point_cloud.get_shs,
            "active_sh_degree": self.active_sh_degree,
            "bg_color": self.background,
        }

        def render_func(data):
            data.update(atributes_dict)
            return self.renderer(**data)

        validation_process(
            render_func,
            self.datapipline,
            self.global_step,
            self.logger
        )

    @torch.no_grad()
    def test(self):
        self.point_cloud.load_ply('/home/clz/code_remote/Pointrix/projects/gaussian_splatting/garden/30000.ply')
        self.point_cloud.to(self.device)
        test_view_render(self.point_cloud, self.renderer, self.datapipline, output_path=self.cfg.output_path)
        novel_view_render(self.point_cloud, self.renderer, self.datapipline, output_path=self.cfg.output_path)
        
    def saving(self):
        data_list = {
            "active_sh_degree": self.active_sh_degree,
            "point_cloud": self.point_cloud.state_dict(),
        }
        return data_list