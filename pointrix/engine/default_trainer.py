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
from pointrix.utils.optimizer import parse_scheduler, parse_optimizer
from pointrix.point_cloud import parse_point_cloud

from torch.utils.tensorboard import SummaryWriter

class DefaultTrainer:
    @dataclass
    class Config:
        # Modules
        point_cloud: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        renderer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = field(default_factory=dict)
        
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
        self.exp_dir = exp_dir
        self.device = device
        # build config
        self.cfg = parse_structured(self.Config, cfg)
        
        self.start_steps = 1
        self.global_step = 0
        
        self.loop_range = range(self.start_steps, self.cfg.max_steps+1)
        
        # build datapipeline
        self.datapipline = parse_data_pipline(self.cfg.dataset)
        # build renderer
        self.renderer = parse_renderer(self.cfg.renderer)  
        # build model
        self.point_cloud = parse_point_cloud(
            self.cfg.point_cloud, 
            self.datapipline
        )
        
        # Set up scheduler for points cloud position
        self.cameras_extent = self.datapipline.training_dataset.radius
        if self.cfg.scheduler is not None:
            self.schedulers = parse_scheduler(
                self.cfg.scheduler, 
                self.cameras_extent if self.cfg.spatial_lr_scale else 1.
            )
        
        self.setup()    
        # set up optimizer in the end, so that all parameters are registered      
        self.optimizer = parse_optimizer(self.cfg.optimizer, self)
        
        # build logger
        self.logger = SummaryWriter(exp_dir)
        
    def before_train_start(self):
        pass
    
    def train_step(self, batch) -> None:
        raise NotImplementedError
    
    def val_step(self) -> None:
        raise NotImplementedError
    
    def test(self) -> None:
        raise NotImplementedError
    
    def upd_bar_info(self, info: dict) -> None:
        pass
    
    def train_loop(self) -> None:
        self.progress_bar = tqdm(
            range(self.start_steps, self.cfg.max_steps),
            desc="Training progress",
            leave=False,
        )
        bar_info = {}
        self.global_step = self.start_steps
        ema_loss_for_log = 0.0
        self.before_train_start()
        for iteration in self.loop_range:
            self.update_lr()
            batch = self.datapipline.next_train()
            step_dict = self.train_step(batch)
            self.update_state()
            self.optimization()
            
            # Log to progress bar every 10 iterations
            for key, value in step_dict.items():
                if 'loss' in key:
                    ema_loss_for_log = 0.4 * value.item() + 0.6 * ema_loss_for_log
                    bar_info.update({key: f"{ema_loss_for_log:.{7}f}"})
                
                if self.logger:
                    self.logger.add_scalar(key, value, self.global_step)
                    
            if iteration % self.cfg.bar_upd_interval == 0:
                self.upd_bar_info(bar_info)
                self.progress_bar.set_postfix(bar_info)
                self.progress_bar.update(self.cfg.bar_upd_interval)
            
            if iteration % self.cfg.val_interval == 0 or iteration == self.cfg.max_steps:
                self.val_step()
            if iteration % 5000 == 0:
                self.point_cloud.save_ply(os.path.join(self.cfg.output_path, "{}.ply".format(iteration)))
            
            self.global_step += 1
        
        self.progress_bar.close()
    
    def optimization(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
    
    def update_state(self) -> None:
        pass

    def update_lr(self) -> None:
        # Leraning rate scheduler
        if len(self.cfg.scheduler) > 0:
            for param_group in self.optimizer.param_groups:
                name = param_group['name']
                if name in self.schedulers.keys():
                    lr = self.schedulers[name](self.global_step)
                    param_group['lr'] = lr
                    
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
            path = os.path.join(self.cfg.output_path, "chkpnt" + str(self.global_step-1) + ".pth")
        data_list = self.get_saving()
        torch.save(data_list, path)
                
    def load_model(self, path=None) -> None:
        if path is None:
            path = os.path.join(self.cfg.output_path, "chkpnt" + str(self.global_step) + ".pth")
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
    
    
    