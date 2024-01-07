import os
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
from torch import nn

from pointrix.base_model.gaussian_splatting import GaussianSplatting

class GaussianFlow(GaussianSplatting):
    @dataclass
    class Config(GaussianSplatting.Config):
        pos_traj_type: str = 'poly_fourier'
        pos_traj_dim: int = 3
        
        rot_traj_type: str = 'poly_fourier'
        rot_traj_dim: int = 3
        
        scale_traj_type: str = 'none'
        scale_traj_dim: int = 3
        
        opc_traj_type: str = 'none'
        opc_traj_dim: int = 3
        
        feat_traj_type: str = 'none'
        feat_traj_dim: int = 3
        
        feat_rest_traj_type: str = 'none'
        feat_rest_traj_dim: int = 3
        
        traj_init: str = 'zero'
        
        poly_base_factor: float = 1.0
        Hz_base_facto: float = 1.0
        
        rescale_t: bool = True
        rescale_value: float = 1.0
        
        offset_t: bool = True
        offset_value: float = 0.0
        
        normliaze_rot: bool = False
    
    cfg: Config
    
    def setup(self, cfg):
        super().setup(cfg)
        
        # Set up flow paramters
        
        
    def train_step(self):
        pass
    
    def val_step(self):
        pass
    
    def test_step(self):
        pass
    
    def train_loop(self):
        pass
    
    def optimize(self):
        pass