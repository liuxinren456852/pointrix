# Define your own dataset

To add our dataset, we need to inhert BaseReFormatData class and
rewrite `load_camera` and `load_pointcloud`.

First, we need to import base data format from pointrix so that 
we can inherit, registry and modify them.

```python
from pointrix.dataset.base_data import DATA_FORMAT_REGISTRY, BaseReFormatData, SimplePointCloud
from pointrix.camera.camera import Camera
```

Then, we need to implement the function to load the camera in 
dataset. 

```{note} Pointrix support common dataset reading so you do not 
need to implement by yourself in the most cases. There is just an
example to illustrate how to add your own dataset.
```

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22"
:caption: |
:    We *highlight* the modified part.
@DATA_FORMAT_REGISTRY.register()
class NerfiesReFormat(BaseReFormatData):
    def __init__(self,
                 data_root: Path,
                 split: str = 'train',
                 cached_image: bool = True,
                 scale: float = 1.0):
        super().__init__(data_root, split, cached_image, scale)
    
    def load_camera(self, split: str):
        ## load your camera here
        ## for full implemetation, please refer to projects/deformable_gaussian/dataformat.py
        return cameras_results
    
    def load_pointcloud(self):
        xyz = np.load(os.path.join(self.data_root, "points.npy"))
        xyz = (xyz - self.scene_center) * self.coord_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = SimplePointCloud(positions=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))
        return pcd
```

and change the model name in configuiation:

```{code-block} yaml
:lineno-start: 1 
:emphasize-lines: "10"
:caption: |
:    We *highlight* the modified part.
name: "garden"

trainer:
  output_path: "garden_fix"
  max_steps: 30000
  val_interval: 5000

dataset:
    data_path: ""
    data_type: "NerfiesReFormat"
    cached_image: True
    shuffle: True
    batch_size: 1
    num_workers: 0
    scale: 0.25
    white_bg: False

    ...
```
