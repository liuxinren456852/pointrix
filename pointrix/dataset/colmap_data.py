import os
import numpy as np
from typing import Any, Dict, List
from pathlib import Path

from pointrix.camera.camera import Camera
from pointrix.dataset.base_data import BaseReFormatData
from pointrix.dataset.data_utils.colmap_utils import (read_extrinsics_binary, 
                                                      read_intrinsics_binary, 
                                                      qvec2rotmat)


class ColmapReFormat(BaseReFormatData):
    def __init__(self, cfg,
                 data_root: Path,
                 split: str = 'train'):
        super().__init__(cfg, data_root, split)

    def load_camera(self, split: str) -> List[Camera]:
        cameras_extrinsic_file = os.path.join(
            self.data_root, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(
            self.data_root, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        # TODO: more methods for splitting the data
        llffhold = 8
        cameras = []
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = focal_length_x
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            camera = Camera(idx=intr.id, R=R, T=T, width=width, height=height, rgb_file_name=os.path.basename(extr.name),
                            fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, bg=0.0)
            cameras.append(camera)
        if split == 'train':
            cameras_results = [c for idx, c in enumerate(
                sorted(cameras.copy(), key=lambda x: x.rgb_file_name)) if idx % llffhold != 0]
        elif split == 'val':
            cameras_results = [c for idx, c in enumerate(
                sorted(cameras.copy(), key=lambda x: x.rgb_file_name)) if idx % llffhold == 0]
        return cameras_results

    def load_image_filenames(self, cameras: List[Camera], split) -> list[Path]:
        image_filenames = []
        for camera in cameras:
            image_filenames.append(os.path.join(
                self.data_root, "images", camera.rgb_file_name))
        return image_filenames

    def load_metadata(self, split) -> Dict[str, Any]:
        return {}
