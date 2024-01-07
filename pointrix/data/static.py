import os
import sys
import json
from PIL import Image
from pathlib import Path

import torch
import numpy as np

from dataclasses import dataclass
from torch.utils.data import Dataset

from pointrix.utils.config import parse_structured
from pointrix.data.utils import (
    pose_spherical, 
    fov2focal, 
    focal2fov, 
    getNerfppNorm,
    PILtoTorch,
    getProjectionMatrix,
    getWorld2View2
)

def readCamerasFromTransforms(
    path,
    json_file, 
    white_background, 
    extension=".png",
):
    data_list = []
    fovx = json_file["camera_angle_x"]

    frames = json_file["frames"]
    for idx, frame in enumerate(frames):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        
        matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        
        # # NeRF 'transform_matrix' is a camera-to-world transform
        # c2w = np.array(frame["transform_matrix"])
        # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # c2w[:3, 1:3] *= -1

        # # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)
        # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        # T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        image = PILtoTorch(image, (800, 800))
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        
        data_list.append({
            "idx": idx, 
            "R": R, 
            "T": T, 
            "image": image.clamp(0.0, 1.0),
            "image_path": image_path,
            "image_name": image_name,
            "FovX": FovX, 
            "FovY": FovY,
            "width": image.shape[1],
            "height": image.shape[2],
        })
            
    return data_list

class NeRFDataset(Dataset):
    @dataclass
    class Config:
        data_path: str = ""
        extension: str = ".png"
        white_background: bool = True
        
    cfg: Config
    
    def __init__(self, cfg, split):
        
        self.split = split
        self.cfg = parse_structured(self.Config, cfg)
        
        # Read metadata
        train_data, test_data, render_data, nerf_norm = self.read_metadata(self.cfg.data_path)
        self.train_data = train_data
        self.test_data = test_data
        self.render_data = render_data
        self.nerf_norm = nerf_norm
        self.cameras_extent = nerf_norm["radius"]
        
    def __len__(self):
        if self.split == "train":
            return len(self.train_data)
        elif self.split == "test":
            return len(self.test_data)
        elif self.split == "render":
            return len(self.render_data)
        
    def __getitem__(self, index):
        if self.split == "train":
            data = self.train_data[index]
        elif self.split == "test":
            data = self.test_data[index]
        elif self.split == "render":
            data = self.render_data[index]
        
        znear = 100.0
        zfar = 0.01
        
        R = data["R"]
        T = data["T"]
        
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        
        world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(
            znear=znear, 
            zfar=zfar, 
            fovX=data["FovX"], 
            fovY=data["FovY"],
        ).transpose(0,1).cuda()
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(
                projection_matrix.unsqueeze(0)
            )
        ).squeeze(0).cuda()
        camera_center = world_view_transform.inverse()[3, :3].cuda()
        
        resutls = {
            "data_id": index,
            "world_view_transform": world_view_transform,
            "projection_matrix": projection_matrix,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "znear": znear,
            "zfar": zfar,
            "trans": trans,
            "scale": scale,
        }
        resutls.update(data)
        return resutls
        
        
    def read_metadata(self, path):
        with open(os.path.join(path, "transforms_train.json")) as json_file:
            train_json = json.load(json_file)
        with open(os.path.join(path, "transforms_test.json")) as json_file:
            test_json = json.load(json_file)  

        train_data = readCamerasFromTransforms(
            path,
            train_json, 
            self.cfg.white_background, 
            self.cfg.extension, 
        )
        
        test_data = readCamerasFromTransforms(
            path,
            test_json, 
            self.cfg.white_background, 
            self.cfg.extension, 
        )
        
        Rs = []
        Ts = []
        for data in train_data:
            Rs.append(data["R"])
            Ts.append(data["T"])
        
        nerf_norm = getNerfppNorm(Rs, Ts)
        
        # Generate render poses
        render_poses = torch.stack([
            pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,100+1)[:-1]], 0)
        
        fovx = train_data[0]["FovX"]
        image = train_data[0]["image"]
        
        render_data = []
        for idx, poses in enumerate(render_poses):
            matrix = np.linalg.inv(np.array(poses))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx
            render_data.append({
                "idx": idx, 
                "R": R, 
                "T": T, 
                "FovX": FovX, 
                "FovY": FovY,
                "width": image.shape[1],
                "height": image.shape[2],
            })

        return train_data, test_data, render_data, nerf_norm