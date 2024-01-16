import os
from tqdm import tqdm
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import torch
from torch import nn
from pointrix.renderer import parse_renderer
from pointrix.utils.config import parse_structured
from pointrix.dataset.base_data import BaseDataPipline
from pointrix.utils.optimizer import parse_scheduler, parse_optimizer

from torch.utils.tensorboard import SummaryWriter

class DefaultTrainer(nn.Module):
    @dataclass
    class Config:
        # Modules
        points_cloud: dict = field(default_factory=dict)
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
        
        # Progress bar
        bar_upd_interval: int = 10
        
        # Output path
        output_path: str = "output"
        
    cfg: Config
        
    def __init__(self, cfg, exp_dir, device="cuda") -> None:
        super().__init__()
        self.exp_dir = exp_dir
        self.device = device
        self.cfg = parse_structured(self.Config, cfg)
        
        self.datapipline = BaseDataPipline(self.cfg.dataset)
        if self.cfg.scheduler is not None:
            self.schedulers = parse_scheduler(self.cfg.scheduler)
            
        self.renderer = parse_renderer(self.cfg.renderer)  
        # all trainers should implement setup
        self.setup(self.datapipline.point_cloud)    
        # set up optimizer in the end, so that all parameters are registered      
        self.optimizer = parse_optimizer(self.cfg.optimizer, self)
        
        self.start_steps = 1
        self.global_step = 0
        
        self.loop_range = range(self.start_steps, self.cfg.max_steps+1)
        self.progress_bar = tqdm(
            range(self.start_steps, self.cfg.max_steps),
            desc="Training progress",
            leave=False,
        )
        
        self.progress_bar_info = {}
        
        self.logger = SummaryWriter(exp_dir)
    
    def train_step(self, batch) -> None:
        raise NotImplementedError
    
    def val_step(self) -> None:
        raise NotImplementedError
    
    def test(self) -> None:
        raise NotImplementedError
    
    def upd_bar_info(self, info: dict) -> None:
        pass
    
    def train_loop(self) -> None:
        bar_info = self.progress_bar_info
        self.global_step = self.start_steps
        ema_loss_for_log = 0.0
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
            
            if iteration % self.cfg.val_interval == 0:
                self.val_step()
            
            self.global_step += 1
        
        self.progress_bar.close()
    
    def optimization(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
    
    def update_state(self) -> None:
        pass

    def update_lr(self) -> None:
        # Leraning rate scheduler
        if self.cfg.scheduler is not None:
            for param_group in self.optimizer.param_groups:
                name = param_group['name']
                if name in self.schedulers.keys():
                    lr = self.schedulers[name](self.global_step)
                    param_group['lr'] = lr
                    
    def save_model(self, path=None) -> None:
        if path is None:
            path = os.path.join(self.cfg.output_path, "chkpnt" + str(self.global_step) + ".pth")
        torch.save(self.state_dict(), path)
                
    
    
    