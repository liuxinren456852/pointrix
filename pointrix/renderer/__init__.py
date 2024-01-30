from .splatting import GaussianSplattingRender, RENDERER_REGISTRY

def parse_renderer(cfg, **kwargs):
    return RENDERER_REGISTRY.get(cfg.name)(cfg, **kwargs)