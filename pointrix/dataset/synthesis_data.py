from pathlib import Path
from typing import List
from pointrix.camera.camera import Camera
from pointrix.dataset.base_data import BaseDataFormat, BaseReFormatData, DATA_FORMAT_REGISTRY, SimplePointCloud, BaseImageDataset, BaseDataPipeline
from pointrix.utils.dataset.dataset_utils import fov2focal, focal2fov
from pointrix.utils.dataset.synthesis_utils import *
from pointrix.camera.camera import Cameras
import numpy as np
from pointrix.utils.sh_utils import SH2RGB
import os
from dataclasses import dataclass, field
from pointrix.utils.config import parse_structured
from plyfile import PlyData
from pointrix.utils.dataset.colmap_utils import retrieve_ply_file, save_ply_file

import torchvision.transforms as T

import imageio


@DATA_FORMAT_REGISTRY.register()
class SynthesisReFormat(BaseReFormatData):
    @dataclass
    class Config:
        base_name: str = "base40M-textvec"
        prompt: str = "a red motorcycle"
        radius_range: List[float] = field(default_factory=lambda: [0.16, 0.60])
        max_radius_range: List[float] = field(
            default_factory=lambda: [3.5, 5.0])
        default_radius: float = 3.5
        theta_range: List[float] = field(default_factory=lambda: [45, 105])
        max_theta_range: List[float] = field(default_factory=lambda: [45, 105])
        phi_range: List[float] = field(default_factory=lambda: [-180, 180])
        max_phi_range: List[float] = field(default_factory=lambda: [-180, 180])
        fovy_range: List[float] = field(default_factory=lambda: [0.32, 0.60])
        max_fovy_range: List[float] = field(
            default_factory=lambda: [0.16, 0.60])
        rand_cam_gamma: float = 1.0
        angle_overhead: int = 30
        angle_front: int = 60
        render_45: bool = True
        uniform_sphere_rate: float = 0
        image_w: int = 512
        image_h: int = 512
        SSAA: float = 1.0
        init_num_pts: int = 4096
        default_polar: int = 90
        default_azimuth: int = 0
        default_fovy: float = 0.55  # 20
        device: str = "cuda"
        jitter_pose: bool = True
        jitter_center: float = 0.05
        jitter_target: float = 0.05
        jitter_up: float = 0.01
        use_pointe_rgb: bool = False
        image_count: int = 200
        path: str = "./point_cloud/"
        init_shape: str = "pointe"
        generate_size: int = 4
        fov: float = 0.48
        validation_size: int = 120
        loss:dict =field(default_factory=dict)
    def __init__(self,
                 data_root: Path,
                 split: str = "train",
                 scale: float = 1.0,
                 cached_image: bool = True,
                 gencfg: dict = {}
                 ):
        self.cfg = parse_structured(self.Config, gencfg)
        super().__init__(data_root, split, cached_image, scale, gencfg)

    def load_pointcloud(self) -> SimplePointCloud:
        ply_path = os.path.join(self.data_root, "sparse/0/points3D.ply")
        ply_path=os.path.join("store_point_cloud", "pointe.ply")
        if not os.path.exists(ply_path):

            num_pts = self.cfg.init_num_pts
            if self.cfg.init_shape == 'sphere':
                thetas = np.random.rand(num_pts)*np.pi
                phis = np.random.rand(num_pts)*2*np.pi
                radius = np.random.rand(num_pts)*0.5
                # We create random points inside the bounds of sphere
                xyz = np.stack([
                    radius * np.sin(thetas) * np.sin(phis),
                    radius * np.sin(thetas) * np.cos(phis),
                    radius * np.cos(thetas),
                ], axis=-1)  # [B, 3]
            elif self.cfg.init_shape == 'box':
                xyz = np.random.random((num_pts, 3)) * 1.0 - 0.5
            elif self.cfg.init_shape == 'rectangle_x':
                xyz = np.random.random((num_pts, 3))
                xyz[:, 0] = xyz[:, 0] * 0.6 - 0.3
                xyz[:, 1] = xyz[:, 1] * 1.2 - 0.6
                xyz[:, 2] = xyz[:, 2] * 0.5 - 0.25
            elif self.cfg.init_shape == 'rectangle_z':
                xyz = np.random.random((num_pts, 3))
                xyz[:, 0] = xyz[:, 0] * 0.8 - 0.4
                xyz[:, 1] = xyz[:, 1] * 0.6 - 0.3
                xyz[:, 2] = xyz[:, 2] * 1.2 - 0.6
            elif self.cfg.init_shape == 'pointe':
                num_pts = int(num_pts/5000)

                xyz, rgb = init_by_point_e(self.cfg.base_name, self.cfg.prompt,num_pts)
                xyz[:, 1] = - xyz[:, 1]
                xyz[:, 2] = xyz[:, 2] + 0.15
                thetas = np.random.rand(num_pts)*np.pi
                phis = np.random.rand(num_pts)*2*np.pi
                radius = np.random.rand(num_pts)*0.05
                # We create random points inside the bounds of sphere
                xyz_ball = np.stack([
                    radius * np.sin(thetas) * np.sin(phis),
                    radius * np.sin(thetas) * np.cos(phis),
                    radius * np.cos(thetas),
                ], axis=-1)  # [B, 3]expend_dims
                rgb_ball = np.random.random((4096, num_pts, 3))*0.0001
                rgb = (np.expand_dims(rgb, axis=1)+rgb_ball).reshape(-1, 3)
                xyz = (np.expand_dims(xyz, axis=1) +
                       np.expand_dims(xyz_ball, axis=0)).reshape(-1, 3)
                xyz = xyz * 1.
                num_pts = xyz.shape[0]
            elif self.cfg.init_shape == 'scene':
                thetas = np.random.rand(num_pts)*np.pi
                phis = np.random.rand(num_pts)*2*np.pi
                radius = np.random.rand(num_pts) + self.cfg.radius_range[-1]*3
                # We create random points inside the bounds of sphere
                xyz = np.stack([
                    radius * np.sin(thetas) * np.sin(phis),
                    radius * np.sin(thetas) * np.cos(phis),
                    radius * np.cos(thetas),
                ], axis=-1)  # [B, 3]
            else:
                raise NotImplementedError()
            print(f"Generating random point cloud ({num_pts})...")

            shs = np.random.random((num_pts, 3)) / 255.0

            if self.cfg.init_shape == 'pointe' and self.cfg.use_pointe_rgb:
                pcd = SimplePointCloud(
                    positions=xyz, colors=rgb, normals=np.zeros((num_pts, 3)))
                save_ply_file(ply_path, xyz, rgb * 255)
            else:
                pcd = SimplePointCloud(positions=xyz, colors=SH2RGB(
                    shs), normals=np.zeros((num_pts, 3)))
                save_ply_file(ply_path, xyz, SH2RGB(shs) * 255)

        positions, colors, normals = retrieve_ply_file(ply_path)

        return SimplePointCloud(positions=positions, colors=colors, normals=normals)

    def load_data_list(self, split) -> BaseDataFormat:
        camera, _ = self.load_camera(split=split)
        image_filenames = self.load_image_filenames(camera, split=split)
        metadata = self.load_metadata(split=split)
        pointcloud = self.load_pointcloud()
        data = BaseDataFormat(image_filenames, camera,
                              PointCloud=pointcloud, metadata=metadata)

        self.Camera_list = data.Camera_list
        self.images = data.images
        self.image_filenames = data.image_filenames

        return data

    def load_camera(self, split) -> List[Camera]:
        if split == 'train':
            cameras, spherical_coordinate = generate_random_cameras(
                self.cfg['generate_size'], self.cfg, SSAA=self.cfg.SSAA)
        elif split == "val":
            cameras, spherical_coordinate = generate_circle_cameras(
                self.cfg, self.cfg['validation_size'], self.cfg['render_45'])
        elif split == "test":
            cameras, spherical_coordinate = generate_circle_cameras(
                self.cfg, self.cfg['validation_size'], self.cfg['render_45'])
        return cameras, spherical_coordinate

    def load_image_filenames(self, cameras: List[Camera], split) -> list[Path]:
        """
        The function for loading the image files names typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        image_filenames = []
        for camera in cameras:
            image_filenames.append(camera.rgb_file_name)
        return image_filenames

    def load_data_list_include_camera(self):
        split = self.split
        camera, spherical_coordinate = self.load_camera(split)
        image_filenames = self.load_image_filenames(camera, split=split)
        data = BaseDataFormat(
            image_filenames, camera, PointCloud=self.data_list.PointCloud, metadata=self.data_list.metadata)
        return data, spherical_coordinate


class SynthesisImageDataset(BaseImageDataset):
    def __init__(self, format_data: SynthesisReFormat) -> None:
        super().__init__(format_data)
        self.format_data = format_data
        self.len = len(self.camera_list)

    def __len__(self):
        self.resample()
        return self.len

    def resample(self):
        format_data, spherical_coordinate = self.format_data.load_data_list_include_camera()
        self.spherical_coordinate = spherical_coordinate
        self.camera_list = format_data.Camera_list
        self.images = format_data.images
        self.image_file_names = format_data.image_filenames

        self.cameras = Cameras(self.camera_list)
        self.radius = self.cameras.radius

    def __getitem__(self, idx):
        # self.resample()
        camera = self.camera_list[idx]
        image = None
        camera.height = camera.height
        camera.width = camera.width
        return {
            "image": image,
            "camera": camera,
            "FovX": camera.fovX,
            "FovY": camera.fovY,
            "height": camera.image_height,
            "width": camera.image_width,
            "world_view_transform": camera.world_view_transform,
            "full_proj_transform": camera.full_proj_transform,
            "camera_center": camera.camera_center,
            "polar": self.spherical_coordinate["polar"][idx],
            "azimuth": self.spherical_coordinate["azimuth"][idx],
            "camera_distances": self.spherical_coordinate['radius'][idx]
        }


class SynthesisImageDataPipeline(BaseDataPipeline):
    def __init__(self, cfg: BaseDataPipeline.Config, dataformat) -> None:
        self.cfg = parse_structured(self.Config, cfg)
        self._fully_initialized = True

        self.train_format_data = dataformat(
            data_root=self.cfg.data_path, split="train",
            cached_image=self.cfg.cached_image,
            scale=self.cfg.scale,
            gencfg=self.cfg.generate_cfg)
        self.validation_format_data = dataformat(
            data_root=self.cfg.data_path, split="val",
            cached_image=self.cfg.cached_image,
            scale=self.cfg.scale,
            gencfg=self.cfg.generate_cfg)

        self.point_cloud = self.train_format_data.data_list.PointCloud
        self.white_bg = self.cfg.white_bg
        self.use_dataloader = self.cfg.use_dataloader

        # assert not self.use_dataloader and self.cfg.batch_size == 1 and \
        #     self.cfg.cached_image, "Currently only support batch_size=1, cached_image=True when use_dataloader=False"
        self.loaddata()

        self.training_cameras = self.training_dataset.cameras

    def get_training_dataset(self) -> BaseImageDataset:
        self.training_dataset = SynthesisImageDataset(
            format_data=self.train_format_data)

    def get_validation_dataset(self) -> BaseImageDataset:
        self.validation_dataset = SynthesisImageDataset(
            format_data=self.validation_format_data
        )
