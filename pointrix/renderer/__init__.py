from .splatting import splatting_render

def parse_renderer(cfg):
    if cfg.name == 'splatting':
        return splatting_render
    else:
        raise NotImplementedError