# Trainer

The Trainer provides a complete default training pipeline, including **initialization** of hooks, data pipelines, renderers, models, optimizers, and loggers, as well as the **train_step**, **train_loop**, and **validation** processes.

```python
class DefaultTrainer:
    """
    The default trainer class for training and testing the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    exp_dir : str
        The experiment directory.
    device : str, optional
        The device to use, by default "cuda".
    """
    @dataclass
    class Config:
        # Modules
        model: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        renderer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = field(default_factory=dict)
        writer: dict = field(default_factory=dict)
        hooks: dict = field(default_factory=dict)
        # Dataset
        dataset_name: str = "NeRFDataset"
        dataset: dict = field(default_factory=dict)

        # Training config
        batch_size: int = 1
        num_workers: int = 0
        max_steps: int = 30000
        val_interval: int = 2000
        spatial_lr_scale: bool = True

        # Progress bar
        bar_upd_interval: int = 10
        # Output path
        output_path: str = "output"

    cfg: Config

    def __init__(self, cfg: Config, exp_dir: Path, device: str = "cuda") -> None:
        super().__init__()
        self.exp_dir = exp_dir
        self.device = device

        self.start_steps = 1
        self.global_step = 0

        # build config
        self.cfg = parse_structured(self.Config, cfg)
        # build hooks
        self.hooks = parse_hooks(self.cfg.hooks)
        self.call_hook("before_run")
        # build datapipeline
        self.datapipeline = parse_data_pipeline(self.cfg.dataset)

        # build render and point cloud model
        self.white_bg = self.datapipeline.white_bg
        self.renderer = parse_renderer(
            self.cfg.renderer, white_bg=self.white_bg, device=device)

        self.model = parse_model(
            self.cfg.model, self.datapipeline, device=device)

        # build optimizer and scheduler
        cameras_extent = self.datapipeline.training_dataset.radius
        self.schedulers = parse_scheduler(self.cfg.scheduler,
                                          cameras_extent if self.cfg.spatial_lr_scale else 1.
                                          )
        self.optimizer = parse_optimizer(self.cfg.optimizer,
                                         self.model,
                                         cameras_extent=cameras_extent)

        # build logger and hooks
        self.logger = parse_writer(self.cfg.writer, exp_dir)

    def train_step(self, batch: List[dict]) -> None:
        """
        The training step for the model.

        Parameters
        ----------
        batch : dict
            The batch data.
        """


    @torch.no_grad()
    def validation(self):
        """
        The validation progress.
        """
        self.val_dataset_size = len(self.datapipeline.validation_dataset)
        for i in range(0, self.val_dataset_size):
            self.call_hook("before_val_iter")
            batch = self.datapipeline.next_val(i)
            render_dict = self.model(batch)
            render_results = self.renderer.render_batch(render_dict, batch)
            self.metric_dict = self.model.get_metric_dict(render_results, batch)
            self.call_hook("after_val_iter")

    def test(self, model_path=None) -> None:
        """
        The testing method for the model.
        """

    def train_loop(self) -> None:
        """
        The training loop for the model.
        """
        

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """
        Call the hook method.

        Parameters
        ----------
        fn_name : str
            The hook method name.
        kwargs : dict
            The keyword arguments.
        """
        for hook in self.hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None
        
```

You can also define your own training process by define hook function or inherit the DefaultTrainer class and add your modifcation.

More details can be found in API and hook part.