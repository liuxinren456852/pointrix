from .base_data import DATA_FORMAT_REGISTRY
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))

for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
    mod = __import__('.'.join([__name__, py]), fromlist=[py])
    classes = [getattr(mod, x)
               for x in dir(mod) if isinstance(getattr(mod, x), type)]
    for cls in classes:
        setattr(sys.modules[__name__], cls.__name__, cls)


def parse_data_pipline(cfg):
    if len(cfg) == 0:
        return None
    data_type = cfg.data_type
    dataformat = DATA_FORMAT_REGISTRY.get(data_type)

    return BaseDataPipline(cfg, dataformat)
