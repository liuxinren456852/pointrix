import torch
from torch import nn
from pytorch_msssim import ms_ssim
from dataclasses import dataclass

from pointrix.point_cloud import PointCloud, POINTSCLOUD_REGISTRY
from pointrix.utils.losses import l1_loss, l2_loss, ssim
from .gaussian_utils import (
    build_covariance_from_scaling_rotation,
    inverse_sigmoid,
    gaussian_point_init,
    psnr
)

from lpips import LPIPS
lpips_net = LPIPS(net="vgg").to("cuda")
def lpips_norm_fn(x): return x[None, ...] * 2 - 1
def lpips_norm_b_fn(x): return x * 2 - 1
def lpips_fn(x, y): return lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()


def lpips_b_fn(x, y): return lpips_net(
    lpips_norm_b_fn(x), lpips_norm_b_fn(y)).mean()

@POINTSCLOUD_REGISTRY.register()
class GaussianPointCloud(PointCloud):
    @dataclass
    class Config(PointCloud.Config):
        max_sh_degree: int = 3
        lambda_dssim: float = 0.2
        
    cfg: Config
    
    def setup(self, point_cloud=None):
        super().setup(point_cloud)
        # Activation funcitons
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        scales, rots, opacities, features_rest = gaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
        )

        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )

        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)
        
    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self.scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)

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
            self.features,self.features_rest,
        ], dim=1)

    @property
    def get_position(self):
        return self.position
    
    def forward(self, batch=None):
        render_dict = {
            "position": self.position,
            "opacity": self.get_opacity,
            "scaling": self.get_scaling,
            "rotation": self.get_rotation,
            "shs": self.get_shs,
        }
        return render_dict
    
    def get_loss_dict(self, render_results, batch):
        gt_images = torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))], 
            dim=0
        )
        L1_loss = l1_loss(render_results['images'], gt_images)
        ssim_loss = 1.0 - ssim(render_results['images'], gt_images)
        loss = (
            (1.0 - self.cfg.lambda_dssim) * L1_loss
        ) + (
            self.cfg.lambda_dssim * ssim_loss
        )
        loss_dict = {"loss": loss,
                    "L1_loss": L1_loss,
                    "ssim_loss": ssim_loss}
        return loss_dict
    
    def get_optimizer_dict(self, loss_dict, render_results, white_bg):
        optimizer_dict = {"loss": loss_dict["loss"],
                        "viewspace_points": render_results['viewspace_points'],
                        "visibility": render_results['visibility'],
                        "radii": render_results['radii'],
                        "white_bg":white_bg}
        return optimizer_dict
    
    def get_metric_dict(self, render_results, batch):
        gt_images = torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))], 
            dim=0
        )
        L1_loss = l1_loss(render_results['images'], gt_images)
        psnr_test = psnr(render_results['images'], gt_images).mean().double()
        ssims_test = ms_ssim(
            render_results['images'], gt_images, data_range=1, size_average=True
        )

        lpips_test = lpips_fn(render_results['images'].squeeze(), gt_images.squeeze()).item()
        metric_dict = {"L1_loss": L1_loss,
                    "psnr": psnr_test,
                    "ssims": ssims_test,
                    "lpips": lpips_test, 
                    "gt_images": gt_images, 
                    "images": render_results['images'],
                    "rgb_file_name": batch[0]["camera"].rgb_file_name}
        return metric_dict