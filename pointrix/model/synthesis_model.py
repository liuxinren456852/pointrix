from .base_model import MODEL_REGISTRY
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from threestudio.models.guidance.stable_diffusion_guidance import StableDiffusionGuidance
from threestudio.models.guidance.stable_diffusion_vsd_guidance import StableDiffusionVSDGuidance
from threestudio.models.guidance.stable_diffusion_unified_guidance import StableDiffusionUnifiedGuidance
# from threestudio.utils.misc import C
import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Union
from omegaconf import DictConfig
from pytorch_msssim import ms_ssim

from pointrix.utils.config import parse_structured
from pointrix.point_cloud import parse_point_cloud
from .loss import l1_loss, ssim, psnr,tv_loss
from pointrix.utils.registry import Registry
from pointrix.utils.config import C
from PIL import Image


from .base_model import BaseModel
import sys
# sys.path.insert(0,'../')
sys.path.insert(0, "/home/linxi/Pointrix/threestudio")
# from threestudio.threestudio.models.guidance.stable_diffusion_guidance


@MODEL_REGISTRY.register()
class SynthesisModel(BaseModel):
    """
    class for Synthesis models.

    Parameters
    ----------
    cfg : Optional[Union[dict, DictConfig]]
        The configuration dictionary.
    datapipeline : BaseDataPipeline
        The data pipeline which is used to initialize the point cloud.
    device : str, optional
        The device to use, by default "cuda".
    """
    
    @dataclass
    class Syn_Config:

        sd_cfg: dict = field(default_factory=dict)
        prompt_cfg: dict = field(default_factory=dict)
        prompt_name: str = 'StableDiffusionPromptProcessor'
        loss:dict = field(default_factory=dict)
    def __init__(self, cfg: Optional[Union[dict, DictConfig]], datapipline, device="cuda"):
        synthesis_cfg = cfg['synthesis_cfg']
        super().__init__(cfg, datapipline, device)
        cfg = parse_structured(self.Syn_Config, synthesis_cfg)
        self.global_step=0
        self.sd_cfg = synthesis_cfg['sd_cfg']
        self.prompt_cfg = synthesis_cfg['prompt_cfg']
        self.loss_cfg=synthesis_cfg['loss']
        self.guidance = StableDiffusionGuidance(self.sd_cfg)
        if cfg['prompt_name'] == 'StableDiffusionPromptProcessor':
            self.prompt_utils = StableDiffusionPromptProcessor(self.prompt_cfg)
        
    def C(self,value,epoch=0):
        
        return C(value,epoch,self.global_step)

    def get_loss_dict(self, render_results, batch, **kwargs) -> dict:
        """
        Get the loss dictionary.

        Parameters
        ----------
        render_results : dict
            The render results which is the output of the renderer.
        batch : dict
            The batch of data which contains the ground truth images.
        
        Returns
        -------
        dict
            The loss dictionary which contain loss for backpropagation.
        """
        
        self.global_step=kwargs['global_step']
        render_dict=kwargs['render_dict']
        rgb = render_results["rgb"].permute(0, 2, 3, 1)
        depth=render_results["depth"].permute(0, 2, 3, 1)
        length=len(batch)
        azimuth =torch.tensor ([batch[idx]['azimuth'] for idx in range(length) ])
        elevation = torch.tensor([batch[idx]['polar'] for idx in range(length)]) 
        camera_distances = torch.tensor([batch[idx]['camera_distances'] for idx in range(length)]) 

        prompt_utils = self.prompt_utils()
        rgb_as_latents = False
        guidance_eval = False

        guidance_out = self.guidance(
            rgb, prompt_utils,  azimuth, elevation, camera_distances, rgb_as_latents, guidance_eval, **kwargs)
        
        
        loss=guidance_out['loss_sds']*self.loss_cfg['lambda_sds']
        if self.loss_cfg["lambda_position"] > 0.0:
            xyz_mean =render_dict['position'].norm(dim=-1)
            loss_position = xyz_mean.mean()
            loss += self.C(self.loss_cfg["lambda_position"]) * loss_position

        if self.loss_cfg["lambda_opacity"] > 0.0:
            scaling = render_dict['scaling'].norm(dim=-1)
            loss_opacity = (
                scaling.detach().unsqueeze(-1) * render_dict['opacity']
            ).sum()
            loss += self.C(self.loss_cfg["lambda_opacity"]) * loss_opacity

        if self.loss_cfg["lambda_scales"] > 0.0:
            scale_sum = torch.sum(render_dict['scaling'])
            loss += self.C(self.loss_cfg["lambda_scales"]) * scale_sum

        if self.loss_cfg["lambda_tv_loss"] > 0.0:
            loss_tv_rgb = self.C(self.loss_cfg["lambda_tv_loss"]) * tv_loss(
                rgb.permute(0, 3, 1, 2)
            )
            loss += loss_tv_rgb
            
        if self.loss_cfg["lambda_depth_tv_loss"] > 0.0:
            loss_tv_depth= self.C(self.loss_cfg["lambda_depth_tv_loss"]) * tv_loss(
                depth.permute(0, 3, 1, 2)
            )
            loss += loss_tv_depth
    
        loss_dict = {"loss": loss}
        if "loss_sds_img" in guidance_out.keys():
            loss_dict.update({"loss_sds_img": guidance_out["loss_sds_img"]})
        self.loss_dict = loss_dict
        return loss_dict

    def get_optimizer_dict(self, loss_dict, render_results, white_bg) -> dict:
        """
        Get the optimizer dictionary which will be 
        the input of the optimizer update model

        Parameters
        ----------
        loss_dict : dict
            The loss dictionary.
        render_results : dict
            The render results which is the output of the renderer.
        white_bg : bool
            The white background flag.
        """
        
        optimizer_dict = {"loss": loss_dict["loss"],
                          "viewspace_points": render_results['viewspace_points'],
                          "visibility": render_results['visibility'],
                          "radii": render_results['radii'],
                          "white_bg": white_bg}
        return optimizer_dict

    def get_metric_dict(self, render_results, batch) -> dict:
        """
        Get the metric dictionary.

        Parameters
        ----------
        render_results : dict
            The render results which is the output of the renderer.
        batch : dict
            The batch of data which contains the ground truth images.
        
        Returns
        -------
        dict
            The metric dictionary which contains the metrics for evaluation.
        """
        
        metric_dict = {"loss": self.loss_dict['loss'],
                       "images": render_results['images']}
        if "loss_sds_img" in self.loss_dict.keys():
            metric_dict.update({"loss_sds_img":self.loss_dict['loss_sds_img']})

        return metric_dict
