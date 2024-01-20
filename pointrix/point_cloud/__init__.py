import os
import sys

path = os.path.dirname(os.path.abspath(__file__))

for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)

from .points_gaussian import GAUSSIAN_REGISTRY

def build_gaussian(cfg, datapipline):
    
    if len(cfg) == 0:
        return None
    gaussian_type = cfg.gaussian_type
    gaussian = GAUSSIAN_REGISTRY.get(gaussian_type)
    assert gaussian is not None, "Gaussian is not registered: {}".format(
        gaussian_type
    )
    return gaussian(cfg, datapipline.point_cloud, datapipline.training_dataset.radius)