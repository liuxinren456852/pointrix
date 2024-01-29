import os
import random
from tqdm import tqdm
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import torch
from torch import nn
from pointrix.renderer import parse_renderer
from pointrix.dataset import parse_data_pipline
from pointrix.utils.config import parse_structured
from pointrix.utils.optimizer import parse_scheduler
from pointrix.optimizer import parse_optimizer
from pointrix.point_cloud import parse_point_cloud
from pointrix.logger import parse_writer, create_progress
from pointrix.hook import parse_hooks

from torch.utils.tensorboard import SummaryWriter


class DefaultTrainer:
    @dataclass
    class Config:
        # Modules
        point_cloud: dict = field(default_factory=dict)
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

    def __init__(self, cfg, exp_dir, device="cuda") -> None:
        super().__init__()
        
        # build config
        self.cfg = parse_structured(self.Config, cfg)
        # build datapipeline
        self.datapipline = parse_data_pipline(self.cfg.dataset)
        # build renderer
        self.renderer = parse_renderer(self.cfg.renderer)
        # build model
        self.point_cloud = parse_point_cloud(
            self.cfg.point_cloud,
            self.datapipline).to(device)

        # Set up scheduler for points cloud position
        cameras_extent = self.datapipline.training_dataset.radius
        if self.cfg.scheduler is not None:
            self.schedulers = parse_scheduler(
                self.cfg.scheduler,
                cameras_extent if self.cfg.spatial_lr_scale else 1.
            )

        # set up optimizer in the end, so that all parameters are registered
        self.optimizer = parse_optimizer(self.cfg.optimizer,
                                         self.point_cloud,
                                         cameras_extent=cameras_extent)
        # build logger
        self.writter = parse_writer(self.cfg.writer, exp_dir)
        self.logger = SummaryWriter(exp_dir)
        self.hooks = parse_hooks(self.cfg.hooks)

        self.exp_dir = exp_dir
        self.device = device
        self.active_sh_degree = 0

        # TODO: use camera to get the extent
        self.white_bg = self.datapipline.white_bg

        bg_color = [1, 1, 1] if self.white_bg else [0, 0, 0]
        self.background = torch.tensor(
            bg_color,
            dtype=torch.float32,
            device=self.device,
        )
        self.start_steps = 1
        self.global_step = 0
        self.call_hook("before_run")

    def train_step(self, batch) -> None:
        raise NotImplementedError

    def val_step(self) -> None:
        raise NotImplementedError

    def test(self) -> None:
        raise NotImplementedError

    def train_loop(self) -> None:
        loop_range = range(self.start_steps, self.cfg.max_steps+1)
        self.progress_bar = tqdm(
            range(self.start_steps, self.cfg.max_steps),
            desc="Training progress",
            leave=False,
        )
        self.global_step = self.start_steps
        for iteration in loop_range:
            self.call_hook("before_train_iter")
            batch = self.datapipline.next_train()
            self.loss_dict, self.optimizer_dict = self.train_step(batch)
            self.optimizer.update_model(self.optimizer_dict)
            self.call_hook("after_train_iter")
            self.global_step += 1
            if iteration % self.cfg.val_interval == 0 or iteration == self.cfg.max_steps:
                self.val_step()
        self.progress_bar.close()

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            **kwargs: Keyword arguments passed to hook.
        """
        for hook in self.hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None

    def saving_base(self):
        data_list = {
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
        }
        return data_list

    def saving(self):
        pass

    def get_saving(self):
        data_list = self.saving_base()
        data_list.update(self.saving())
        return data_list

    def save_model(self, path=None) -> None:
        if path is None:
            path = os.path.join(self.cfg.output_path,
                                "chkpnt" + str(self.global_step-1) + ".pth")
        data_list = self.get_saving()
        torch.save(data_list, path)

    def load_model(self, path=None) -> None:
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
