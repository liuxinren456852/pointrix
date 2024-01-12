import torch
from pathlib import Path
from abc import abstractmethod
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Any, Dict, Union, List

from pointrix.camera.camera import Camera, TrainableCamera
from pointrix.dataset.data_utils.dataset_utils import force_full_init


@dataclass
class BaseDataFormat:
    image_filenames: List[Path]
    """camera image filenames"""
    Cameras: List[Camera]
    """camera parameters"""
    metadata: Dict[str, Any] = field(default_factory=lambda: dict({}))
    """other information that is required for the dataset"""

    def __getitem__(self, item):
        return self.image_filenames[item], self.Cameras[item]

    def __len__(self):
        return len(self.image_filenames)


class BaseReFormatData:
    def __init__(self, config,
                 data_root: Path,
                 split: str = "train"):
        self.config = config
        self.data_root = data_root
        self.split = split
        self.data_list = self.load_data_list(self.split)

    def load_data_list(self, split) -> BaseDataFormat:
        camera = self.load_camera(split=split)
        image_filenames = self.load_image_filenames(camera, split=split)
        metadata = self.load_metadata(split=split)
        data = BaseDataFormat(image_filenames, camera, metadata)
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


class BaseDataset(Dataset):
    def __init__(self, config, format_data):
        self.config = config
        self.format_data = format_data

    def __len__(self):
        return len(self.format_data)

    def __getitem__(self, idx):
        pass


class BaseDataPipline:
    def __init__(self, config):
        self._fully_initialized = False
        self.config = config

    # TODO use rigistry
    @abstractmethod
    def get_training_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def get_validation_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def loaddata(self):
        self.training_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )

        self.iter_train_image_dataloader = iter(self.training_loader)
        self.iter_val_image_dataloader = iter(self.validation_loader)

    @abstractmethod
    def next_train(self, step=int):
        raise NotImplementedError

    @abstractmethod
    def next_val(self, step=int):
        raise NotImplementedError

    @abstractmethod
    def get_param_groups(self):
        raise NotImplementedError
