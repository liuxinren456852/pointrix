# Example

There is a simple example that how to extend our pointrix 
framework to dynamic gaussaian, which the deformation is generated
by MLP which take time step as input.


## Add your model

First, we need to import base model from pointrix so that 
we can inherit, registry and modify them.

```python
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY
```

Then, we need to implemet our model based BaseModel which 
contains full gaussian point implemetation.

```{note} Your can refer to pointrix/model/base_model.py for more detail if
    you care about the full gaussian point implemetation.
```

```{code-block} python
:lineno-start: 1 
:emphasize-lines: "8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 45"
:caption: |
:    We *highlight* the modified part.
@MODEL_REGISTRY.register()
class DeformGaussian(BaseModel):
    def __init__(self, cfg, datapipline, device="cuda"):
        super().__init__(cfg, datapipline, device)

        # you can refer to projects/deformable_gaussian/model.py
        # if you want to know the detail of DeformNetwork.
        self.deform = DeformNetwork(is_blender=False).to(self.device)

        # The gaussian point cloud is implemeted in BaseModel, we
        # do not need to care about the detail here.
    
    def forward(self, batch):
        camera_fid = torch.Tensor([batch[0]['camera'].fid]).float().to(self.device)
        position = self.point_cloud.get_position
        time_input = camera_fid.unsqueeze(0).expand(position.shape[0], -1)
        d_xyz, d_rotation, d_scaling = self.deform(position, time_input)

        render_dict = {
            "position": self.point_cloud.position + d_xyz,
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling + d_scaling,
            "rotation": self.point_cloud.get_rotation + d_rotation,
            "shs": self.point_cloud.get_shs,
        }
        
        return render_dict
    
    def get_param_groups(self):
        param_group = {}
        param_group[self.point_cloud.prefix_name +
                    'position'] = self.point_cloud.position
        param_group[self.point_cloud.prefix_name +
                    'opacity'] = self.point_cloud.opacity
        param_group[self.point_cloud.prefix_name +
                    'features'] = self.point_cloud.features
        param_group[self.point_cloud.prefix_name +
                    'features_rest'] = self.point_cloud.features_rest
        param_group[self.point_cloud.prefix_name +
                    'scaling'] = self.point_cloud.scaling
        param_group[self.point_cloud.prefix_name +
                    'rotation'] = self.point_cloud.rotation


        param_group['deform'] = self.deform.parameters()
        return param_group
```

## Add your dataset

To add our dataset, we need to inhert BaseReFormatData class and
rewrite `load_camera` and `load_pointcloud`.

First, we need to import base data format from pointrix so that 
we can inherit, registry and modify them.

```python
from pointrix.dataset.base_data import DATA_FORMAT_REGISTRY, BaseReFormatData, BasicPointCloud
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
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))
        return pcd
```

## Run pointrix
We can add our model and dataset implemented above in 
pointrix framework.

```{note} we need to modify the model and dataset name 
to our names of new model and datasets implemented above.
Pointrix framework can find your model and datasets by registry.
```

```{code-block} python
:lineno-start: 1  # this is a comment
: # this is also a comment
:emphasize-lines: "1, 2, 8, 9, 10, 11, 12, 13, 14"
:caption: |
:    We *highlight* the modified part.

from model import DeformGaussian
from dataformat import NerfiesReFormat

def main(args, extras) -> None:
    
    cfg = load_config(args.config, cli_args=extras)

    cfg.trainer.model.name = "DeformGaussian"
    cfg.trainer.dataset.data_type = "NerfiesReFormat"
    # you need to modify this path to your dataset.
    cfg.trainer.dataset.data_path = "/home/clz/data/dnerf/cat"
    cfg['trainer']['optimizer']['optimizer_1']['params']['deform'] = {}
    cfg['trainer']['optimizer']['optimizer_1']['params']['deform']['lr'] = 0.00016 * 5.0
    cfg.trainer.val_interval = 5000

    gaussian_trainer = DefaultTrainer(
        cfg.trainer,
        cfg.exp_dir,
    )
    gaussian_trainer.train_loop()    
    model_path = os.path.join(
        cfg.exp_dir, 
        "chkpnt" + str(gaussian_trainer.global_step) + ".pth"
    )
    gaussian_trainer.save_model(model_path)
    print("\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default = None)
    args, extras = parser.parse_known_args()
    
    main(args, extras)
```

Finally, you can run the command to run your code.

```bash
python launch.py --config default.yaml
```
