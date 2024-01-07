import os
from tqdm import tqdm
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import torch
from torch import nn
from pointrix.renderer import parse_renderer
from pointrix.utils.config import parse_structured
from pointrix.data.dataloader import parse_dataloader
from pointrix.utils.optimizer import parse_scheduler, parse_optimizer


class DefaultTrainer(nn.Module):
    @dataclass
    class Config:
        # Modules
        points_cloud: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        renderer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = field(default_factory=dict)
        
        # Dataset 
        dataset: dict = field(default_factory=dict)
        
        # Training config
        max_steps: int
        val_interval: int
        test_interval: int
        
        # Progress bar
        bar_upd_interval: int = 10
        
    cfg: Config
        
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        # all trainers should implement setup
        self.setup()
        
        self.renderer = parse_renderer(self.cfg.renderer)
        
        if self.cfg.scheduler is not None:
            self.schedulers = parse_scheduler(self.cfg.scheduler)
            
        self.optimizer = parse_optimizer(self.cfg.optimizer, self)
        self.dataloader = parse_dataloader(self.cfg.dataset)
        
        self.start_steps = 0
        self.global_step = 0
        
        self.loop_range = range(self.start_steps, self.cfg.max_steps)
        self.progress_bar = tqdm(self.loop_range, desc="Training progress")
        
        self.progress_bar_info = {}
    
    def train_step(self, batch) -> None:
        raise NotImplementedError
    
    def val_step(self, batch) -> None:
        raise NotImplementedError
    
    def test(self) -> None:
        raise NotImplementedError
    
    def upd_bar_info(self, info: dict) -> None:
        pass
    
    def train_loop(self) -> None:
        bar_info = self.progress_bar_info
        self.train_loader = iter(self.dataloader["train"])
        for iteration in self.loop_range:
            try:
                batch = next(self.train_loader)
            except StopIteration:
                self.train_loader = iter(self.dataloader["train"])
            
            step_dict = self.train_step(batch)
            self.global_step += 1
            self.update_state()
            self.optimization()
            
            # Log to progress bar every 10 iterations
            for key, value in step_dict.items():
                if 'loss' in key:
                    ema_loss_for_log = 0.4 * value.item() + 0.6 * ema_loss_for_log
                    bar_info.update({key: f"{ema_loss_for_log:.{7}f}"})
                    
            if iteration % self.cfg.bar_upd_interval == 0:
                self.upd_bar_info(bar_info)
                self.progress_bar.set_postfix(bar_info)
                self.progress_bar.update(self.cfg.bar_upd_interval)
                
            if iteration == self.cfg.max_steps:
                self.progress_bar.close()
            
            if iteration % self.cfg.val_interval:
                for batch_val in self.dataloader["val"]:
                    self.val_step(batch_val)
    
    def optimization(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
    
    def update_state(self) -> None:
        # Leraning rate scheduler
        if self.cfg.scheduler is not None:
            for param_group in self.optimizer.param_groups:
                name = param_group['name']
                if name in self.schedulers.keys():
                    lr = self.schedulers[name](self.global_step)
                    param_group[name] = lr
                    
    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)
                
    
    
    