# Model

## Point Cloud

We have defined a PointCloud class that supports flexible definition and rendering of various attributes. For example, if you wish to add depth as an attribute to the point cloud, you can do so as follows:

```python
point_cloud = PointsCloud(cfg)
point_cloud.register_atribute('depth', depth)
```

Based on Point Cloud, we further define the Gaussian Point Class, which contains  `position`, `feature`, `feature_rest`, `opacity`, `scale` by default.
And we also define some custome operation on Gaussian Points such as clone, split, remove and reset, which should be combined with Gaussian Optimizers.

## Base model
The base model includes the following interfaces that need to be defined:

```{note}
You can refer to the API part for more details of BaseModel.
Only little part need to be modified if you want add your model in Pointrix.
Tutorial illustrates how to add your own model in Pointrix.
```
```python
@MODEL_REGISTRY.register()
class BaseModel(BaseModule):
    """
    Base class for all models.

    Parameters
    ----------
    cfg : Optional[Union[dict, DictConfig]]
        The configuration dictionary.
    datapipeline : BaseDataPipeline
        The data pipeline which is used to initialize the point cloud.
    device : str, optional
        The device to use, by default "cuda".
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, datapipeline, device="cuda"):
        self.point_cloud = parse_point_cloud(self.cfg.point_cloud,
                                             datapipeline).to(device)
        self.point_cloud.set_prefix_name("point_cloud")
        self.device = device

    def forward(self, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """

    def get_loss_dict(self, render_results, batch) -> dict:
        """
        Get the loss dictionary.

        Parameters
        ----------
        render_results : dict
            The render results which is the output of the renderer.
        batch : dict
            The batch of data which contains the ground truth images.
        
        Returns
        -------
        dict
            The loss dictionary which contain loss for backpropagation.
        """

    def get_optimizer_dict(self, loss_dict, render_results, white_bg) -> dict:
        """
        Get the optimizer dictionary which will be 
        the input of the optimizer update model

        Parameters
        ----------
        loss_dict : dict
            The loss dictionary.
        render_results : dict
            The render results which is the output of the renderer.
        white_bg : bool
            The white background flag.
        """

    @torch.no_grad()
    def get_metric_dict(self, render_results, batch) -> dict:
        """
        Get the metric dictionary.

        Parameters
        ----------
        render_results : dict
            The render results which is the output of the renderer.
        batch : dict
            The batch of data which contains the ground truth images.
        
        Returns
        -------
        dict
            The metric dictionary which contains the metrics for evaluation.
        """
    
    def load_ply(self, path):
        """
        Load the ply model for point cloud.

        Parameters
        ----------
        path : str
            The path of the ply file.
        """

```

## Optimizer

An optimizer is a component responsible for executing parameter updates through gradient descent in a model. Pointrix further encapsulates optimizers at the level of PyTorch optimizers.

Compared to optimizers that only update model parameters, we propose **Gaussian Optimizers**. As depicted in the diagram below, these optimizers not only update model parameters but also adjust the number, position, and size of point clouds.

```{image} ../images/gs_optimizer.svg
:class: sd-animate-grow50-rot20
:align: center
```