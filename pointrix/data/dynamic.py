import os
import sys
import json
from PIL import Image
from pathlib import Path

import torch
import numpy as np

from dataclasses import dataclass
from torch.utils.data import Dataset

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
    mapper = {}, 
    factor=1.0
):
    data_list = []
    fovx = json_file["camera_angle_x"]

    frames = json_file["frames"]
    for idx, frame in enumerate(frames):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        time = mapper[frame["time"]]
        matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        image = PILtoTorch(image, (int(800 * factor), int(800 * factor)))
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
            "timestamp": time, 
            "FovX": FovX, 
            "FovY": FovY,
            "width": image.shape[1],
            "height": image.shape[2],
        })
            
    return data_list

class DNeRFDataset(Dataset):
    @dataclass
    class Config:
        data_path: str = ""
        factor: float = 1.0
        extension: str = ".png"
        white_background: bool = True
        
    cfg: Config
    
    def __init__(self, cfg, split):
        
        self.split = split
        self.cfg = cfg
        
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
        
        trans = torch.tensor([0.0, 0.0, 0.0])
        scale = 1.0
        
        world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=znear, 
            zfar=zfar, 
            fovX=data["FovX"], 
            fovY=data["FovY"],
        ).transpose(0,1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(
                projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        
        results = {
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
        
        results.update(data)
        
        return results
        
        
    def read_metadata(self, path):
        with open(os.path.join(path, "transforms_train.json")) as json_file:
            train_json = json.load(json_file)
        with open(os.path.join(path, "transforms_test.json")) as json_file:
            test_json = json.load(json_file)  
        time_line = (
            [frame["time"] for frame in train_json["frames"]]
        ) + (
            [frame["time"] for frame in test_json["frames"]]
        )
        time_line = set(time_line)
        time_line = list(time_line)
        time_line.sort()
        timestamp_mapper = {}
        max_time_float = max(time_line)
        for index, time in enumerate(time_line):
            # timestamp_mapper[time] = index
            timestamp_mapper[time] = time/max_time_float
            
        train_data = readCamerasFromTransforms(
            path,
            train_json, 
            self.cfg.white_background, 
            self.cfg.extension, 
            timestamp_mapper, 
            self.cfg.factor
        )
        
        test_data = readCamerasFromTransforms(
            path,
            test_json, 
            self.cfg.white_background, 
            self.cfg.extension, 
            timestamp_mapper, 
            self.cfg.factor
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
        render_times = torch.linspace(
            0, max_time_float, self.render_poses.shape[0]
        )
        
        fovx = train_data[0]["FovX"]
        image = train_data[0]["image"]
        
        render_data = []
        for idx, (time, poses) in enumerate(zip(render_times, render_poses)):
            time = time/max_time_float
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
                "timestamp": time, 
                "FovX": FovX, 
                "FovY": FovY,
                "width": image.shape[1],
                "height": image.shape[2],
            })

        return train_data, test_data, render_data, nerf_norm