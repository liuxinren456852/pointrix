import torch
import numpy as np
from PIL import Image
from torch import Tensor
from pathlib import Path
from jaxtyping import Float
from abc import abstractmethod
from torch.utils.data import Dataset
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Union, List, NamedTuple, Optional

from pointrix.utils.config import parse_structured
from pointrix.camera.camera import Camera, TrainableCamera
from pointrix.dataset.utils.dataset_utils import force_full_init, getNerfppNorm


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


@dataclass
class BaseDataFormat:
    image_filenames: List[Path]
    """camera image filenames"""
    Cameras: List[Camera]
    """camera image list"""
    images: Optional[List[Image.Image]] = None
    """camera parameters"""
    PointCloud: Union[BasicPointCloud, None] = None
    """precompute pointcloud"""
    metadata: Dict[str, Any] = field(default_factory=lambda: dict({}))
    """other information that is required for the dataset"""

    def __getitem__(self, item) -> tuple[Path, Camera]:
        return self.image_filenames[item], self.Cameras[item]

    def __len__(self) -> int:
        return len(self.image_filenames)


class BaseReFormatData:

    def __init__(self, data_root: Path,
                 split: str = "train",
                 cached_image: bool = True):
        self.data_root = data_root
        self.split = split
        self.data_list = self.load_data_list(self.split)

        if cached_image:
            self.data_list.images = self.load_all_images()

    def load_data_list(self, split) -> BaseDataFormat:
        camera = self.load_camera(split=split)
        image_filenames = self.load_image_filenames(camera, split=split)
        metadata = self.load_metadata(split=split)
        data = BaseDataFormat(image_filenames, camera, metadata=metadata)
        return data

    @abstractmethod
    def load_camera(self, split) -> List[Camera]:
        raise NotImplementedError

    @abstractmethod
    def load_image_filenames(self, split) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def load_metadata(self, split) -> Dict[str, Any]:
        raise NotImplementedError

    def load_all_images(self) -> List[Image.Image]:
        return [np.array(Image.open(image_filename), dtype="uint8")
                for image_filename in self.data_list.image_filenames]


# TODO: support different dataset (Meta information (depth) support)
class BaseImageDataset(Dataset):
    def __init__(self, format_data: BaseDataFormat) -> None:
        self.cameras = format_data.Cameras
        self.images = format_data.images
        self.image_file_names = format_data.image_filenames
        Rs, Ts = [], []
        for camera in self.cameras:
            Rs.append(camera.R)
            Ts.append(camera.T)
        Rs = torch.stack(Rs, dim=0)
        Ts = torch.stack(Ts, dim=0)
        self.radius = getNerfppNorm(Rs.numpy(), Ts.numpy())["radius"]

        if self.images is not None:
            self.images = [self._transform_image(
                image) for image in self.images]

    # TODO: full init
    def __len__(self):
        return len(self.cameras)

    def _transform_image(self, image, bg=[1., 1., 1.]):
        image = image / 255.
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.shape[2] in [
            3, 4], f"Image shape of {image.shape} is in correct."

        if image.shape[2] == 4:
            image = image[:, :, :3] * image[:, :, 3:4] + \
                bg * (1 - image[:, :, 3:4])
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        return image.clamp(0.0, 1.0)

    def __getitem__(self, idx):
        image_file_name = self.image_file_names[idx]
        camera = self.cameras[idx]
        image = self._load_transform_image(
            image_file_name) if self.images is None else self.images[idx]
        camera.height = image.shape[1]
        camera.width = image.shape[2]
        return {"image": image, "camera": camera}

    def _load_transform_image(self, image_filename, bg=[1., 1., 1.]) -> Float[Tensor, "3 h w"]:
        pil_image = np.array(Image.open(image_filename),
                             dtype="uint8")

        return self._transform_image(pil_image)


class BaseDataPipline:
    @dataclass
    class Config:
        # Datatype
        data_path: str = "data"
        data_type: str = "nerf_synthetic"
        cached_image: bool = True
        shuffle: bool = True
        batch_size: int = 1
        num_workers: int = 1
        white_bg: bool = False
    cfg: Config

    def __init__(self, cfg) -> None:
        self.cfg = parse_structured(self.Config, cfg)
        self._fully_initialized = True

        # TODO: use registry
        if self.cfg.data_type == "colmap":
            from pointrix.dataset.colmap_data import ColmapReFormat as ReFormat
        elif self.cfg.data_type == "nerf_synthetic":
            from pointrix.dataset.nerf_data import NerfReFormat as ReFormat

        self.train_format_data = ReFormat(
            data_root=self.cfg.data_path, split="train", cached_image=self.cfg.cached_image).data_list
        self.validation_format_data = ReFormat(
            data_root=self.cfg.data_path, split="val", cached_image=self.cfg.cached_image).data_list

        self.point_cloud = self.train_format_data.PointCloud
        self.white_bg = self.cfg.white_bg
        self.loaddata()

    # TODO use rigistry
    def get_training_dataset(self) -> BaseImageDataset:
        # TODO: use registry
        self.training_dataset = BaseImageDataset(
            format_data=self.train_format_data)

    def get_validation_dataset(self) -> BaseImageDataset:
        self.validation_dataset = BaseImageDataset(
            format_data=self.validation_format_data)

    def loaddata(self) -> None:
        self.get_training_dataset()
        self.get_validation_dataset()

        self.training_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=list
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=list
        )
        self.iter_train_image_dataloader = iter(self.training_loader)
        self.iter_val_image_dataloader = iter(self.validation_loader)

    def next_train(self, step: int = -1):
        try:
            return next(self.iter_train_image_dataloader)
        except StopIteration:
            self.iter_train_image_dataloader = iter(self.training_loader)
            return next(self.iter_train_image_dataloader)

    def next_val(self, step: int = -1):
        try:
            return next(self.iter_val_image_dataloader)
        except StopIteration:
            self.iter_val_image_dataloader = iter(self.validation_loader)
            return next(self.iter_val_image_dataloader)

    @property
    def training_dataset_size(self) -> int:
        return len(self.training_dataset)

    @property
    def validation_dataset_size(self) -> int:
        return len(self.validation_dataset)

    def get_param_groups(self) -> Any:
        raise NotImplementedError
