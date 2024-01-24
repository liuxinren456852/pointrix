import math
import torch
from torch import Tensor
from torch import nn
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from typing import Union, List
from dataclasses import dataclass, field
# from pointrix.camera.camera_utils import se3_exp_map
from pointrix.utils.pose import se3_exp_map

from .camera import Camera


class Cameras:
    def __init__(self, camera_list: List(Camera)):
        self.cameras = camera_list
        self.num_cameras = len(camera_list)
        self.Rs = torch.stack([cam.R for cam in camera_list], dim=0)
        self.Ts = torch.stack([cam.T for cam in camera_list], dim=0)
        self.Ks = torch.stack([cam.K for cam in camera_list], dim=0)
        self.projection_matrices = torch.stack(
            [cam.projection_matrix for cam in camera_list], dim=0)
        self.world_view_transforms = torch.stack(
            [cam.world_view_transform for cam in camera_list], dim=0)
        self.full_proj_transforms = torch.stack(
            [cam.full_proj_transform for cam in camera_list], dim=0)
        self.camera_centers = torch.stack(
            [cam.camera_center for cam in camera_list], dim=0)  # (N, 3)
        
        self.translate, self.radius = self.get_translate_radius()

    def __len__(self):
        return self.num_cameras

    def __getitem__(self, index):
        return self.cameras[index]

    def get_translate_radius(self):
        avg_cam_center = torch.mean(self.cam_centers, dim=0, keepdims=True)
        dist = torch.linalg.norm(
            self.cam_centers - avg_cam_center, dim=1, keepdims=True)
        diagonal = torch.max(dist)
        center = avg_cam_center[0]
        radius = diagonal * 1.1
        translate = -center

        return translate, radius
