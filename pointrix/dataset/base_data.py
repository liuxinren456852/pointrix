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

from pointrix.utils.registry import Registry
from pointrix.utils.config import parse_structured
from pointrix.camera.camera import Camera, Cameras, TrainableCamera
from pointrix.dataset.utils.dataset_utils import force_full_init, getNerfppNorm

DATA_FORMAT_REGISTRY = Registry("DATA_FORMAT", modules=["pointrix.dataset"])
DATA_FORMAT_REGISTRY.__doc__ = ""


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


@dataclass
class BaseDataFormat:
    """
    Pointrix standard data format in Datapipeline.

    Parameters
    ----------
    image_filenames: List[Path]
        The filenames of the images in data.
    Cameras: List[Camera]
        The camera parameters of the images in data.
    images: Optional[List[Image.Image]] = None
        The images in data, which are only needed when cached image is enabled in dataset.
    PointCloud: Union[BasicPointCloud, None] = None
        The pointclouds of the scene, which are used to initialize the gaussian model, enabling better results.
    metadata: Dict[str, Any] = field(default_factory=lambda: dict({}))
        Other information that is required for the dataset.

    Notes
    -----
    1. The order of all data needs to be consistent.
    2. The length of all data needs to be consistent.

    Examples
    --------
    >>> data = BaseDataFormat(image_filenames, camera, metadata=metadata)

    """

    image_filenames: List[Path]
    """camera image filenames"""
    Camera_list: List[Camera]
    """camera image list"""
    images: Optional[List[Image.Image]] = None
    """camera parameters"""
    PointCloud: Union[BasicPointCloud, None] = None
    """precompute pointcloud"""
    metadata: Dict[str, Any] = field(default_factory=lambda: dict({}))
    """other information that is required for the dataset"""

    def __getitem__(self, item) -> tuple[Path, Camera]:
        return self.image_filenames[item], self.Camera_list[item]

    def __len__(self) -> int:
        return len(self.image_filenames)


class BaseReFormatData:
    """
    The foundational classes for formating the data.

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

    def __init__(self, data_root: Path,
                 split: str = "train",
                 cached_image: bool = True,
                 scale: float = 1.0):
        self.data_root = data_root
        self.split = split
        self.scale = scale
        self.data_list = self.load_data_list(self.split)

        if cached_image:
            self.data_list.images = self.load_all_images()

    def load_data_list(self, split) -> BaseDataFormat:
        """
        The foundational function for formating the data

        Parameters
        ----------
        split: The split of the data.
        """
        camera = self.load_camera(split=split)
        image_filenames = self.load_image_filenames(camera, split=split)
        metadata = self.load_metadata(split=split)
        data = BaseDataFormat(image_filenames, camera, metadata=metadata)
        return data

    @abstractmethod
    def load_camera(self, split) -> List[Camera]:
        """
        The function for loading the camera typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        raise NotImplementedError

    @abstractmethod
    def load_image_filenames(self, split) -> list[Path]:
        """
        The function for loading the image files names typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        raise NotImplementedError

    @abstractmethod
    def load_metadata(self, split) -> Dict[str, Any]:
        """
        The function for loading other information that is required for the dataset typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        raise NotImplementedError

    def load_all_images(self) -> List[Image.Image]:
        """
        The function for loading cached images typically requires user customization.

        Parameters
        ----------
        split: The split of the data.
        """
        image_lists = []
        for image_filename in self.data_list.image_filenames:
            temp_image = Image.open(image_filename)
            w, h = temp_image.size
            resize_image = temp_image.resize((
                int(w * self.scale),
                int(h * self.scale)
            ))
            image_lists.append(
                np.array(resize_image, dtype="uint8")
            )
        return image_lists

# TODO: support different dataset (Meta information (depth) support)


class BaseImageDataset(Dataset):
    """
    Basic dataset used in Datapipeline.

    Parameters
    ----------
    format_data: BaseDataFormat
        the standard data format in Datapipeline.

    Notes
    -----
    Cameras and image_file_names are necessary.

    """

    def __init__(self, format_data: BaseDataFormat) -> None:
        self.camera_list = format_data.Camera_list
        self.images = format_data.images
        self.image_file_names = format_data.image_filenames

        self.cameras = Cameras(self.camera_list)
        self.radius = self.cameras.radius

        if self.images is not None:
            self.images = [self._transform_image(
                image) for image in self.images]

    # TODO: full init
    def __len__(self):
        return len(self.camera_list)

    def _transform_image(self, image, bg=[1., 1., 1.]):
        """
        Transform image to torch tensor and normalize to [0, 1]

        Parameters
        ----------
        image: np.array()
            The image to be transformed.
        bg: List[float]
            The background color.
        """
        image = image / 255.
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.shape[2] in [
            3, 4], f"Image shape of {image.shape} is in correct."

        if image.shape[2] == 4:
            image = image[:, :, :3] * image[:, :, 3:4] + \
                bg * (1 - image[:, :, 3:4])
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        return image.clamp(0.0, 1.0)

    def __getitem__(self, idx):
        image_file_name = self.image_file_names[idx]
        camera = self.camera_list[idx]
        image = self._load_transform_image(
            image_file_name) if self.images is None else self.images[idx]
        camera.height = image.shape[1]
        camera.width = image.shape[2]
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
        }

    def _load_transform_image(self, image_filename, bg=[1., 1., 1.]) -> Float[Tensor, "3 h w"]:
        """
        Load image, transform it to torch tensor and normalize to [0, 1]

        Parameters
        ----------
        image: np.array()
            The image to be transformed.
        bg: List[float]
            The background color.
        """
        pil_image = np.array(Image.open(image_filename),
                             dtype="uint8")

        return self._transform_image(pil_image, bg)


class BaseDataPipeline:
    """
    Basic Pipline used in Pointrix

    Parameters
    ----------
    cfg: Config
        the configuration of the dataset.
    dataformat: standard data format in Pointrix


    Notes
    -----
    BaseDataPipeline is always called by build_data_pipline

    """
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
        scale: float = 1.0
    cfg: Config

    def __init__(self, cfg: Config, dataformat) -> None:
        self.cfg = parse_structured(self.Config, cfg)
        self._fully_initialized = True

        self.train_format_data = dataformat(
            data_root=self.cfg.data_path, split="train",
            cached_image=self.cfg.cached_image,
            scale=self.cfg.scale).data_list
        self.validation_format_data = dataformat(
            data_root=self.cfg.data_path, split="val",
            cached_image=self.cfg.cached_image,
            scale=self.cfg.scale).data_list

        self.point_cloud = self.train_format_data.PointCloud
        self.white_bg = self.cfg.white_bg
        self.loaddata()

        self.training_cameras = self.training_dataset.cameras

    # TODO use rigistry
    def get_training_dataset(self) -> BaseImageDataset:
        """
            Return training dataset
        """
        # TODO: use registry
        self.training_dataset = BaseImageDataset(
            format_data=self.train_format_data)

    def get_validation_dataset(self) -> BaseImageDataset:
        """
        Return validation dataset
        """
        self.validation_dataset = BaseImageDataset(
            format_data=self.validation_format_data)

    def loaddata(self) -> None:
        """
        Load dataset into dataloader.
        """
        self.get_training_dataset()
        self.get_validation_dataset()

        self.training_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=list,
            pin_memory=False
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=list,
            pin_memory=False
        )
        self.iter_train_image_dataloader = iter(self.training_loader)
        self.iter_val_image_dataloader = iter(self.validation_loader)

    def next_train(self, step: int = -1) -> Any:
        """
        Generate batch data for trainer

        Parameters
        ----------
        cfg: step
            the training step in trainer.
        """
        try:
            return next(self.iter_train_image_dataloader)
        except StopIteration:
            self.iter_train_image_dataloader = iter(self.training_loader)
            return next(self.iter_train_image_dataloader)

    def next_val(self, step: int = -1) -> Any:
        """
        Generate batch data for validation

        Parameters
        ----------
        cfg: step
            the validation step in validate progress.
        """
        try:
            return next(self.iter_val_image_dataloader)
        except StopIteration:
            self.iter_val_image_dataloader = iter(self.validation_loader)
            return next(self.iter_val_image_dataloader)

    @property
    def training_dataset_size(self) -> int:
        """
        Return training dataset size
        """
        return len(self.training_dataset)

    @property
    def validation_dataset_size(self) -> int:
        """
        Return validation dataset size
        """
        return len(self.validation_dataset)

    def get_param_groups(self) -> Any:
        """
        Return trainable parameters.
        """
        raise NotImplementedError
