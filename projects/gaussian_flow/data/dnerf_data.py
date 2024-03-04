import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

from pointrix.camera.camera import Camera
from pointrix.dataset.base_data import BaseReFormatData, DATA_FORMAT_REGISTRY
from pointrix.utils.dataset.dataset_utils import fov2focal, focal2fov

from .utils import generateCamerasFromTransforms

@dataclass
class CameraTime(Camera):
    timestamp: int = 0
    max_timestamp: int = 0

@DATA_FORMAT_REGISTRY.register()
class DNerfReFormat(BaseReFormatData):
    def __init__(self,
                 data_root: Path,
                 split: str = 'train',
                 scale: float = 1.0,
                 cached_image: bool = True):
        super().__init__(data_root, split, cached_image)

    def load_camera(self, split: str) -> List[Camera]:
        with open(os.path.join(self.data_root, "transforms_train.json")) as json_file:
            train_json = json.load(json_file)
        with open(os.path.join(self.data_root, "transforms_test.json")) as json_file:
            val_json = json.load(json_file)
            
            
        time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in val_json["frames"]]
        max_timestamp = max(time_line)
        
        if split == 'video':
            cameras = generateCamerasFromTransforms(
                self.data_root, 
                "transforms_train.json", "png",
                maxtime=max_timestamp
            )
            return cameras

        if split == 'train':
            json_file = train_json
        elif split == 'val':
            json_file = val_json
            
            
        fovx = json_file["camera_angle_x"]

        frames = json_file["frames"]
        cameras = []
        
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(
                self.data_root, frame["file_path"] + '.png')
            
            timestamp = frame["time"]

            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # R is stored transposed due to 'glm' in CUDA code
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            image_path = os.path.join(self.data_root, cam_name)
            image_name = Path(cam_name).stem

            image = np.array(Image.open(image_path))

            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[0])
            FovY = fovy
            FovX = fovx
            camera = CameraTime(
                idx=idx, R=R, T=T,
                width=image .shape[1], 
                height=image.shape[0],
                rgb_file_name=image_path, 
                fovX=FovX, fovY=FovY, 
                bg=1.0,
                timestamp=timestamp,
                max_timestamp=max_timestamp,
            )
            cameras.append(camera)

        return cameras

    def load_image_filenames(self, cameras: List[Camera], split) -> list[Path]:
        image_filenames = []
        for camera in cameras:
            image_filenames.append(os.path.join(
                self.data_root, "images", camera.rgb_file_name))
        return image_filenames

    def load_metadata(self, split) -> Dict[str, Any]:
        return {}
