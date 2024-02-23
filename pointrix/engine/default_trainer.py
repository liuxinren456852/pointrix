import os
import random
from tqdm import tqdm
from typing import Any, Optional, Union, List
from dataclasses import dataclass, field

import torch
from torch import nn
from pathlib import Path
from pointrix.renderer import parse_renderer
from pointrix.dataset import parse_data_pipeline
from pointrix.utils.config import parse_structured
from pointrix.optimizer import parse_optimizer, parse_scheduler
from pointrix.model import parse_model
from pointrix.logger import parse_writer, create_progress
from pointrix.hook import parse_hooks
from pointrix.exporter.novel_view import test_view_render, novel_view_render

from torch.utils.tensorboard import SummaryWriter


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
        # build datapipeline
        self.datapipline = parse_data_pipeline(self.cfg.dataset)

        # build render and point cloud model
        self.white_bg = self.datapipline.white_bg
        self.renderer = parse_renderer(
            self.cfg.renderer, white_bg=self.white_bg, device=device)

        self.model = parse_model(
            self.cfg.model, self.datapipline, device=device)

        # build optimizer and scheduler
        cameras_extent = self.datapipline.training_dataset.radius
        self.schedulers = parse_scheduler(self.cfg.scheduler,
                                          cameras_extent if self.cfg.spatial_lr_scale else 1.
                                          )
        self.optimizer = parse_optimizer(self.cfg.optimizer,
                                         self.model,
                                         cameras_extent=cameras_extent)

        # build logger and hooks
        self.logger = parse_writer(self.cfg.writer, exp_dir)
        self.hooks = parse_hooks(self.cfg.hooks)

        self.call_hook("before_train")

    def train_step(self, batch: List[dict]) -> None:
        """
        The training step for the model.

        Parameters
        ----------
        batch : dict
            The batch data.
        """
        render_dict = self.model(batch)
        render_results = self.renderer.render_batch(render_dict, batch)
        self.loss_dict = self.model.get_loss_dict(render_results, batch)
        self.optimizer_dict = self.model.get_optimizer_dict(self.loss_dict,
                                                            render_results,
                                                            self.white_bg)

    @torch.no_grad()
    def validation(self):
        self.val_dataset_size = len(self.datapipline.validation_dataset)
        progress_bar = tqdm(
            range(0, self.val_dataset_size),
            desc="Validation progress",
            leave=False,
        )
        for i in range(0, self.val_dataset_size):
            self.call_hook("before_val_iter")
            batch = self.datapipline.next_val(i)
            render_dict = self.model(batch)
            render_results = self.renderer.render_batch(render_dict, batch)
            self.metric_dict = self.model.get_metric_dict(
                render_results, batch)
            self.call_hook("after_val_iter")
            progress_bar.update(1)
        progress_bar.close()
        self.call_hook("after_val")

    def test(self, model_path) -> None:
        """
        The testing method for the model.
        """
        self.model.load_ply(model_path)
        self.model.to(self.device)
        self.renderer.active_sh_degree = 3
        test_view_render(self.model, self.renderer,
                         self.datapipline, output_path=self.cfg.output_path)
        novel_view_render(self.model, self.renderer,
                          self.datapipline, output_path=self.cfg.output_path)

    def train_loop(self) -> None:
        """
        The training loop for the model.
        """
        loop_range = range(self.start_steps, self.cfg.max_steps+1)
        self.progress_bar = tqdm(
            range(self.start_steps, self.cfg.max_steps),
            desc="Training progress",
            leave=False,
        )
        self.global_step = self.start_steps

        self.iter_start = torch.cuda.Event(enable_timing = True)
        self.iter_end = torch.cuda.Event(enable_timing = True)

        for iteration in loop_range:
            
            self.call_hook("before_train_iter")
            
            batch = self.datapipline.next_train(self.global_step)
            self.renderer.update_sh_degree(iteration)
            self.schedulers.step(self.global_step, self.optimizer)
            self.train_step(batch)
            self.optimizer.update_model(**self.optimizer_dict)

            self.call_hook("after_train_iter")
            self.global_step += 1
            if iteration % self.cfg.val_interval == 0 or iteration == self.cfg.max_steps:
                self.validation()
        self.progress_bar.close()
        self.call_hook("after_train")

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

    def load_model(self, path: Path = None) -> None:
        if path is None:
            path = os.path.join(self.cfg.output_path,
                                "chkpnt" + str(self.global_step) + ".pth")
        data_list = torch.load(path)
        self.global_step = data_list["global_step"]
        self.optimizer.load_state_dict(data_list["optimizer"])

        for k, v in data_list.items():
            print(f"Loaded {k} from checkpoint")
            # get arrtibute from model
            arrt = getattr(self, k)
            if isinstance(arrt, nn.Module):
                arrt.load_state_dict(v)
            else:
                setattr(self, k, v)

    def saving(self):
        data_list = {
            "active_sh_degree": self.active_sh_degree,
            "model": self.model.state_dict(),
        }
        return data_list
