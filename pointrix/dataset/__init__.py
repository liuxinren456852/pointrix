import os
import sys

from .base_data import DATA_FORMAT_REGISTRY, BaseDataPipeline
from .colmap_data import ColmapReFormat
from .nerf_data import NerfReFormat
from .synthesis_data import SynthesisReFormat,SynthesisImageDataPipeline


def parse_data_pipeline(cfg: dict):
    """
    Parse the data pipeline.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    if len(cfg) == 0:
        return None
    data_type = cfg.data_type
    dataformat = DATA_FORMAT_REGISTRY.get(data_type)
    if dataformat is SynthesisReFormat:
        return SynthesisImageDataPipeline(cfg,dataformat)
    return BaseDataPipeline(cfg, dataformat)
