
# path = os.path.dirname(os.path.abspath(__file__))

# for py in [f[:-3] for f in os.listdir(path) if f.endswith('.py') and f != '__init__.py']:
#     mod = __import__('.'.join([__name__, py]), fromlist=[py])
#     classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
#     for cls in classes:
#         setattr(sys.modules[__name__], cls.__name__, cls)

from .points import PointCloud

from .points import POINTSCLOUD_REGISTRY

def parse_point_cloud(cfg, datapipline):
    
    if len(cfg) == 0:
        return None
    point_cloud_type = cfg.point_cloud_type
    point_cloud = POINTSCLOUD_REGISTRY.get(point_cloud_type)
    assert point_cloud is not None, "Point Cloud is not registered: {}".format(
        point_cloud_type
    )
    return point_cloud(cfg, datapipline.point_cloud)