# Datapipeline

As shown in the diagram below, the data pipeline consists of three parts: **ReformatData**, **Dataset**, and **DataPipeline**.

- **ReformatData**: Responsible for standardizing user datasets, i.e., converting them to the Pointrix format. If users employ their own datasets, they typically need to inherit this portion of the function.

- **Dataset**: Processes data in the standard format and supports batch size indexing in conjunction with DataLoader. Users usually do not need to overide this part.

- **DataPipeline**: The standard data flow in Pointrix provides a stable data stream for the trainer.

```{image} ../images/data.svg
:class: sd-animate-grow50-rot20
:align: center

    The framework of data pipeline, which user should modify the blue part if employ their own datasets.
```

The data in Pointrix is defined below:
```{note}
If you want to learn the detail of class: **SimplePointCloud**, **Camera** and so on, API
has more details which can help you. 
```

```python
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
    PointCloud: Union[SimplePointCloud, None] = None
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
    PointCloud: Union[SimplePointCloud, None] = None
    """precompute pointcloud"""
    metadata: Dict[str, Any] = field(default_factory=lambda: dict({}))
    """other information that is required for the dataset"""

    def __getitem__(self, item) -> Tuple[Path, Camera]:
        return self.image_filenames[item], self.Camera_list[item]

    def __len__(self) -> int:
        return len(self.image_filenames)
```