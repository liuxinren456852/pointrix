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
        
        self.dataloader = parse_dataloader(
            self.cfg.dataset_name, 
            self.cfg.batch_size,
            self.cfg.num_workers,
            self.cfg.dataset,
        )
        # all trainers should implement setup
        self.setup()
        
        self.renderer = parse_renderer(self.cfg.renderer)
        
        if self.cfg.scheduler is not None:
            self.schedulers = parse_scheduler(self.cfg.scheduler)
            
        self.optimizer = parse_optimizer(self.cfg.optimizer, self)
        
        self.start_steps = 1
        self.global_step = 0
        
        self.loop_range = range(self.start_steps, self.cfg.max_steps+1)
        self.progress_bar = tqdm(self.loop_range, desc="Training progress")
        
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
        self.train_loader = iter(self.dataloader["train"])
        ema_loss_for_log = 0.0
        self.global_step += 1
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
                
                if self.logger:
                    self.logger.add_scalar(key, value, self.global_step)
                    
            if iteration % self.cfg.bar_upd_interval == 0:
                self.upd_bar_info(bar_info)
                self.progress_bar.set_postfix(bar_info)
                self.progress_bar.update(self.cfg.bar_upd_interval)
                
            if iteration == self.cfg.max_steps:
                self.progress_bar.close()
            
            if iteration % self.cfg.val_interval == 0:
                self.val_step()
    
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
                    
    def save_model(self, path=None) -> None:
        if path is None:
            path = os.path.join(self.cfg.output_path, "chkpnt" + str(self.global_step) + ".pth")
        torch.save(self.state_dict(), path)
                
    
    
    