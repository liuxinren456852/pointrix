from .splatting import splatting

def parse_renderer(cfg):
    if cfg['name'] == 'splatting':
        return splatting
    else:
        raise NotImplementedError