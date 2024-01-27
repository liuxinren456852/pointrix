import os
import random
import numpy as np
from tqdm import tqdm
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import torch
import imageio
from torch import nn

from pointrix.utils.losses import l1_loss
from pointrix.utils.system import mkdir_p
from pointrix.model.gaussian_utils import psnr

def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def test_view_render(point_cloud, renderer, datapipeline, output_path, device='cuda'):
    atributes_dict = {
        "position": point_cloud.position,
        "opacity": point_cloud.get_opacity,
        "scaling": point_cloud.get_scaling,
        "rotation": point_cloud.get_rotation,
        "shs": point_cloud.get_shs,
        "active_sh_degree": 3
    }
    atributes_dict['bg_color'] = torch.tensor([
        1., 1., 1.]).float().to(device) if datapipeline.white_bg \
        else torch.tensor([0., 0., 0.]).float().to(device)
    atributes_dict['bg_color'].to(device)
    l1_test = 0.0
    psnr_test = 0.0
    val_dataset = datapipeline.validation_dataset
    val_dataset_size = len(val_dataset)
    progress_bar = tqdm(
        range(0, val_dataset_size),
        desc="Validation progress",
        leave=False,
    )
    renderings = {}
    for i in range(0, val_dataset_size):
        b_i = val_dataset[i]
        atributes_dict.update(b_i)
        image_name = os.path.basename(b_i['camera'].rgb_file_name)
        render_results = renderer(**atributes_dict)
        image = torch.clamp(render_results["render"], 0.0, 1.0)
        gt_image = torch.clamp(b_i['image'].to("cuda").float(), 0.0, 1.0)
        
        mkdir_p(os.path.join(output_path, 'test_view'))
        imageio.imwrite(os.path.join(output_path, 'test_view', image_name),
                        to8b(image.cpu().numpy()).transpose(1, 2, 0))

        l1_test += l1_loss(image, gt_image).mean().double()
        psnr_test += psnr(image, gt_image).mean().double()
        progress_bar.update(1)
    progress_bar.close()
    l1_test /= val_dataset_size
    psnr_test /= val_dataset_size
    print(f"Test results: L1 {l1_test:.5f} PSNR {psnr_test:.5f}")

def novel_view_render(point_cloud, renderer, datapipeline, output_path, novel_view_list=["Dolly", "Zoom", "Spiral"], device='cuda'):
    cameras = datapipeline.training_cameras

    for novel_view in novel_view_list:
        novel_view_camera_list = cameras.generate_camera_path(20, novel_view)
        atributes_dict = {
            "position": point_cloud.position,
            "opacity": point_cloud.get_opacity,
            "scaling": point_cloud.get_scaling,
            "rotation": point_cloud.get_rotation,
            "shs": point_cloud.get_shs,
            "active_sh_degree": 3
        }
        atributes_dict['bg_color'] = torch.tensor([
            1., 1., 1.]).float().to(device) if datapipeline.white_bg \
            else torch.tensor([0., 0., 0.]).float().to(device)
        atributes_dict['bg_color'].to(device)
        for i, camera in enumerate(novel_view_camera_list):
            render_dict = {
                "camera": camera,
                "FovX": camera.fovX,
                "FovY": camera.fovY,
                "height": camera.image_height,
                "width": camera.image_width,
                "world_view_transform": camera.world_view_transform,
                "full_proj_transform": camera.full_proj_transform,
                "camera_center": camera.camera_center,
            }
            render_dict.update(atributes_dict)
            render_results = renderer(**render_dict)
            image = torch.clamp(render_results["render"], 0.0, 1.0)
            mkdir_p(os.path.join(output_path, 'novel_view_' + novel_view))
            imageio.imwrite(os.path.join(output_path, 'novel_view_' + novel_view, "{}.png".format(i)),
                            to8b(image.cpu().numpy()).transpose(1, 2, 0))
        
    

