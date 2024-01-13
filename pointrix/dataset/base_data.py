import torch
import numpy as np
from PIL import Image
from pathlib import Path
from abc import abstractmethod
from torch.utils.data import Dataset
from dataclasses import dataclass, field, asdict
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


## TODO: support cached dataset and lazy init
class BaseImageDataset(Dataset):
    def __init__(self, config, format_data: BaseDataFormat) -> None:
        self.config = config
        self.format_data = format_data

    ## TODO: full init
    def __len__(self):
        return len(self.format_data)

    def __getitem__(self, idx):
        image_file_name, camera = self.format_data[idx]
        image = self.load_image(image_file_name)
        return {"image": image, 
                "camera": asdict(camera)}
    
    def load_image(self, image_filename):
        pil_image = Image.open(image_filename)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)

        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image


class BaseDataPipline:
    def __init__(self, config, path: str):
        self._fully_initialized = False
        self.config = config
        self.path = path

        ## TODO: use registry
        from pointrix.dataset.colmap_data import ColmapReFormat
        self.train_format_data = ColmapReFormat(self.config, data_root=self.path, split="train").data_list
        self.validation_format_data = ColmapReFormat(self.config, data_root=self.path, split="val").data_list

        self.loaddata()

    # TODO use rigistry
    def get_training_dataset(self):
        ## TODO: use registry
        self.training_dataset = BaseImageDataset(config=self.config, format_data=self.train_format_data)
        
    def get_validation_dataset(self):
        self.validation_dataset = BaseImageDataset(config=self.config, format_data=self.validation_format_data)

    def loaddata(self):
        self.get_training_dataset()
        self.get_validation_dataset()

        self.training_loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=5,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=5,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        self.iter_train_image_dataloader = iter(self.training_loader)
        self.iter_val_image_dataloader = iter(self.validation_loader)

    def next_train(self, step=int):
        try:
            return next(self.iter_train_image_dataloader)
        except:
            self.iter_train_image_dataloader = iter(self.training_loader)
            return next(self.iter_train_image_dataloader)

    def next_val(self, step=int):
        return next(self.iter_val_image_dataloader)

    def get_param_groups(self):
        raise NotImplementedError
