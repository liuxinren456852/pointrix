from .gaussian_points.gaussian_points import GaussianPointCloud
from pointrix.dataset.base_data import BaseDataPipeline
from .base_model import BaseModel, MODEL_REGISTRY

__all__ = ["GaussianPointCloud", "BaseModel"]


def parse_model(cfg, datapipline:BaseDataPipeline, device="cuda"):
    """
    Parse the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    datapipline : BaseDataPipeline
        The data pipeline.
    device : str
        The device to use.
    """
    return MODEL_REGISTRY.get(cfg.name)(cfg, datapipline, device)