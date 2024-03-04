import os
import numpy as np
from typing import Any, Dict, List
from pathlib import Path

from pointrix.camera.camera import Camera
from pointrix.dataset.base_data import BaseReFormatData, SimplePointCloud, DATA_FORMAT_REGISTRY
from pointrix.utils.dataset.colmap_utils import (read_extrinsics_binary,
                                                 read_intrinsics_binary,
                                                 fetchPly)
from pointrix.utils.pose import qvec2rotmat
from pointrix.logger.writer import Logger
@DATA_FORMAT_REGISTRY.register()
class ColmapReFormat(BaseReFormatData):
    """
    The foundational classes for formating the colmap data.

    Parameters
    ----------
    data_root: Path
        The root of the data.
    split: str
        The split of the data.
    cached_image: bool
        Whether to cache the image in memory.
    scale: float
        The scene scale of data.
    """
    
    def __init__(self,
                 data_root: Path,
                 split: str = 'train',
                 cached_image: bool = True,
                 scale: float = 1.0):
        super().__init__(data_root, split, cached_image, scale)

    def load_pointcloud(self) -> SimplePointCloud:
        """
        The function for loading the Pointcloud for initialization of gaussian model.
        """
        ply_path = os.path.join(self.data_root, "sparse/0/points3D.ply")
        bin_path = os.path.join(self.data_root, "sparse/0/points3D.bin")
        txt_path = os.path.join(self.data_root, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            from ..utils.dataset.colmap_utils import read_points3D_binary, read_points3D_text, storePly
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        positions, colors, normals = fetchPly(ply_path)
        return SimplePointCloud(positions=positions, colors=colors, normals=normals)

    def load_camera(self, split: str) -> List[Camera]:
        """
        The function for loading the camera typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
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
            height = intr.height * self.scale
            width = intr.width * self.scale

            uid = intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0] * self.scale
                focal_length_y = focal_length_x
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0] * self.scale
                focal_length_y = intr.params[1] * self.scale
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            camera = Camera(idx=idx, R=R, T=T, width=width, height=height, rgb_file_name=os.path.basename(extr.name),
                            fx=focal_length_x, fy=focal_length_y, cx=width/2, cy=height/2, bg=0.0)
            cameras.append(camera)
        
        sorted_camera = sorted(cameras.copy(), key=lambda x: x.rgb_file_name)
        if split == 'train':
            cameras_results = [c for idx, c in enumerate(sorted_camera) if idx % llffhold != 0]
        elif split == 'val':
            cameras_results = [c for idx, c in enumerate(sorted_camera) if idx % llffhold == 0]
        return cameras_results

    def load_image_filenames(self, cameras: List[Camera], split) -> List[Path]:
        """
        The function for loading the image files names typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        image_filenames = []
        for camera in cameras:
            image_filenames.append(os.path.join(
                self.data_root, "images", camera.rgb_file_name))
        return image_filenames

    def load_metadata(self, split) -> Dict[str, Any]:
        """
        The function for loading other information that is required for the dataset typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        meta_data = {}
        if os.path.exists(os.path.join(self.data_root, "depth")):
            depth_folder = "depth" 
        elif os.path.exists(os.path.join(self.data_root, "depths")):
            depth_folder = "depths"
        else:
            depth_folder = None
            Logger.log("No depth folder found, depth will not be loaded.")
        if depth_folder:
            depth_file_names = sorted(os.listdir(os.path.join(self.data_root, depth_folder)))
            meta_data['depth_file_name'] = depth_file_names
        
        return meta_data

            

