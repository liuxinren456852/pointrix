from .splatting import GaussianSplattingRender, RENDERER_REGISTRY

def parse_renderer(cfg, **kwargs):
    """
    Parse the renderer.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    """
    return RENDERER_REGISTRY.get(cfg.name)(cfg, **kwargs)