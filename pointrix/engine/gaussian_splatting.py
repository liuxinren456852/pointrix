import os
import torch
from torch import nn
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict
from pointrix.point_cloud.points_gaussian import GaussianSplatting


from pointrix.engine.trainer import DefaultTrainer
from pointrix.utils.losses import l1_loss, l2_loss, ssim
from pointrix.point_cloud.utils import (
    validation_process,
    render_batch,
)


class GaussianSplattingTrainer(DefaultTrainer):
    @dataclass
    class Config(DefaultTrainer.Config):
        # Train cfg
        lambda_dssim: float = 0.2
        spatial_lr_scale: bool = True

        # Densification
        densify_stop_iter: int = 15000
        densify_start_iter: int = 500
        prune_interval: int = 100
        duplicate_interval: int = 100
        opacity_reset_interval: int = 3000
    cfg: Config

    def setup(self):
        self.gaussian_points = GaussianSplatting(
            self.cfg.gaussian_points, self.datapipline.point_cloud, self.datapipline.training_dataset.radius)

        self.white_bg = self.datapipline.white_bg

        bg_color = [1, 1, 1] if self.white_bg else [0, 0, 0]
        self.background = torch.tensor(
            bg_color,
            dtype=torch.float32,
            device=self.device,
        )

    def saving(self):
        return self.gaussian_points.saving()

    def train_step(self, batch) -> Dict:

        if self.global_step % 1000 == 0:
            self.gaussian_points.update_sh_degree()

        atributes_dict = {
            "position": self.gaussian_points.get_position,
            "opacity": self.gaussian_points.get_opacity,
            "scaling": self.gaussian_points.get_scaling,
            "rotation": self.gaussian_points.get_rotation,
            "shs": self.gaussian_points.get_shs,
            "active_sh_degree": self.gaussian_points.active_sh_degree,
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

        loss.backward()

        self.gaussian_points.accumulate_viewspace_grad(viewspace_points)

        # print("viewspace_grad: ", self.viewspace_grad)
        # TODO: log the learning rate of each elements in optimizer
        for param_group in self.gaussian_points.optimizer.param_groups:
            name = param_group['name']
            if name == "position":
                pos_lr = param_group['lr']
                break

        return {
            "loss": loss,
            "L1_loss": L1_loss,
            "ssim_loss": ssim_loss,
            "num_pt": len(self.gaussian_points.points_cloud),
            "pos_lr": pos_lr,
        }

    def optimization(self) -> None:
        self.gaussian_points.optimizer.step()
        self.gaussian_points.optimizer.zero_grad(set_to_none=True)

    def update_lr(self) -> None:
        # Leraning rate scheduler
        for param_group in self.gaussian_points.optimizer.param_groups:
            name = param_group['name']
            if name in self.gaussian_points.schedulers.keys():
                lr = self.gaussian_points.schedulers[name](self.global_step)
                param_group['lr'] = lr

    def upd_bar_info(self, info: dict) -> None:
        info.update({
            "num_pt": f"{len(self.gaussian_points.points_cloud)}",
        })

    def update_state(self) -> None:
        if self.global_step < self.cfg.densify_stop_iter:
            # Keep track of max radii in image-space for pruning
            visibility = self.visibility
            self.gaussian_points.max_radii2D[visibility] = torch.max(
                self.gaussian_points.max_radii2D[visibility],
                self.radii[visibility]
            )
            self.gaussian_points.pos_gradient_accum[visibility] += torch.norm(
                self.gaussian_points.viewspace_grad[visibility, :2],
                dim=-1,
                keepdim=True
            )
            self.gaussian_points.denom[visibility] += 1
            self.gaussian_points.size_threshold = 20 if self.global_step > self.cfg.opacity_reset_interval else None
            if self.global_step > self.cfg.densify_start_iter:
                # import pdb; pdb.set_trace()
                if self.global_step % self.cfg.duplicate_interval == 0:
                    self.gaussian_points.densify()
                if self.global_step % self.cfg.prune_interval == 0:
                    self.gaussian_points.prune()
                torch.cuda.empty_cache()

            if self.global_step % self.cfg.opacity_reset_interval == 0 or (self.white_bg and self.global_step == self.cfg.densify_start_iter):
                self.gaussian_points.reset_opacity()
        super().update_state()

    @torch.no_grad()
    def val_step(self):
        def render_func(data):
            render_pkg = self.renderer(
                **data,
                position=self.gaussian_points.get_position,
                opacity=self.gaussian_points.get_opacity,
                scaling=self.gaussian_points.get_scaling,
                rotation=self.gaussian_points.get_rotation,
                shs=self.gaussian_points.get_shs,
                active_sh_degree=self.gaussian_points.active_sh_degree,
                bg_color=self.background,
            )
            return render_pkg

        validation_process(
            render_func,
            self.datapipline,
            self.global_step,
            self.logger
        )
