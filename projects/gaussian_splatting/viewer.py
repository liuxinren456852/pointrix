from imgui_bundle import imgui, immvision
import numpy as np
import math
import torch

import os
import argparse
from einops import rearrange

from pointrix.utils.config import load_config
from pointrix.model.gaussian_splatting import GaussianSplatting

ImVec2 = imgui.ImVec2
ImVec4 = imgui.ImVec4


def main(args, extras) -> None:
    from imgui_bundle import immapp

    image = np.zeros((1000, 800, 3), np.uint8)
    h = image.shape[0]
    w = image.shape[1]
    for row in range(h):
        for col in range(w):
            x = col / w * math.pi
            y = row / h * math.pi
            image[row, col, 0] = np.uint8((math.cos(x * 2) + math.sin(y)) * 128)
            image[row, col, 1] = np.uint8((math.cos(x) + math.sin(y * 2)) * 128)
            image[row, col, 2] = np.uint8((math.cos(x * 5) + math.sin(y * 3)) * 128)

    image_params = immvision.ImageParams()
    image_params.image_display_size = (1000, 800)
    
    cfg = load_config(args.config, cli_args=extras)
    gaussian_trainer = GaussianSplatting(
        cfg.trainer,
        cfg.exp_dir,
    )
    model_path = "/home/loyot/workspace/code/Pointrix/projects/gaussian_splatting/output/garden/garden@20240123-023405/chkpnt30001.pth"
    gaussian_trainer.load_model(path=model_path)
    
    val_dataset = gaussian_trainer.datapipline.validation_dataset
    batch_exp = val_dataset[0]
    atributes_dict = {
        "position": gaussian_trainer.point_cloud.position,
        "opacity": gaussian_trainer.point_cloud.get_opacity,
        "scaling": gaussian_trainer.point_cloud.get_scaling,
        "rotation": gaussian_trainer.point_cloud.get_rotation,
        "shs": gaussian_trainer.point_cloud.get_shs,
        "active_sh_degree": gaussian_trainer.active_sh_degree,
        "bg_color": gaussian_trainer.background,
    }
    
    @torch.no_grad()
    def render_func(data):
        data.update(atributes_dict)
        return gaussian_trainer.renderer(**data)
    
    render_results = render_func(batch_exp)
    image = torch.clamp(render_results["render"], 0.0, 1.0)
    image = rearrange(image, "c h w -> h w c")
    h, w = image.shape[0], image.shape[1]
    image_np = image.contiguous().cpu().numpy()
    image_params.image_display_size = (h, w)

    def gui() -> None:
        imgui.text(f"FPS:{imgui.get_io().framerate:.1f}")
        render_results = render_func(batch_exp)
        image = torch.clamp(render_results["render"], 0.0, 1.0)
        image = rearrange(image, "c h w -> h w c")
        h, w = image.shape[0], image.shape[1]
        image_np = image.contiguous().cpu().numpy()
        immvision.image("House", image_np, image_params)

    immapp.run(gui, with_implot=True, with_markdown=True, fps_idle=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default = None)
    args, extras = parser.parse_known_args()
    
    main(args, extras)
