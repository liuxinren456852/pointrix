from .base_splatting import GaussianSplattingRender, RENDERER_REGISTRY
from .dptr import DPTRRender

def parse_renderer(cfg, **kwargs):
    """
    Parse the renderer.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    name = cfg.pop("name")
    return RENDERER_REGISTRY.get(name)(cfg, **kwargs)