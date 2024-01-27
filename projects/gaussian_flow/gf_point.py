import numpy as np

import torch
import torch.nn as nn

from dataclasses import dataclass
from pointrix.model.gaussian_points import GaussianPointCloud
from pointrix.point_cloud import PointCloud, POINTSCLOUD_REGISTRY

import polyfourier


@POINTSCLOUD_REGISTRY.register()
class GaussianFlowPointCloud(GaussianPointCloud):
    @dataclass
    class Config(GaussianPointCloud.Config):
        pos_traj_type: str = 'poly_fourier'
        pos_traj_dim: int = 3
        rot_traj_type: str = 'poly_fourier'
        rot_traj_dim: int = 3
        
        rescale_t: bool = True
        rescale_value: float = 1.0
        
        offset_t: bool = True
        offset_value: float = 0.0
        
        normliaze_rot: bool = False
        normalize_timestamp: bool = False
        
        random_noise: bool = False
        
    cfg: Config
    
    def set_traj_base_dim(self, traj_type, feat_dim, vec_dim):
        if traj_type == 'poly_fourier':
            traj_base_dim = 3
            extend_dim = feat_dim * vec_dim * traj_base_dim
        elif traj_type == 'poly':
            traj_base_dim = 0
            extend_dim = feat_dim * vec_dim
        elif traj_type == 'fourier':
            traj_base_dim = 2
            extend_dim = feat_dim * vec_dim * traj_base_dim
        else:
            raise ValueError(f"Unknown traj_type: {traj_type}")
            
        return traj_base_dim, extend_dim

    def setup(self, point_cloud=None):
        super().setup(point_cloud)
        
        self.rot_traj_base_dim, rot_extend_dim = self.set_traj_base_dim(
            self.cfg.rot_traj_type, self.cfg.rot_traj_dim, 4
        )
            
        rots = torch.zeros((len(self), 4+rot_extend_dim))
        rots[:, 0] = 1
        self.rotation = nn.Parameter(
            rots.contiguous().requires_grad_(True)
        )
            
        self.rot_fit_model = polyfourier.get_fit_model(type_name=self.cfg.rot_traj_type)

        
        # init position trajectory
        self.pos_traj_base_dim, pos_extend_dim = self.set_traj_base_dim(
            self.cfg.pos_traj_type, self.cfg.pos_traj_dim, 3
        )
            
        self.position = nn.Parameter(
            torch.cat([
                self.position,
                torch.zeros(
                    (len(self), pos_extend_dim),
                    dtype=torch.float32
                )
            ], dim=1).contiguous().requires_grad_(True)
        )
        self.pos_fit_model = polyfourier.get_fit_model(type_name=self.cfg.pos_traj_type)
        
        self.register_atribute("time_center", torch.randn((len(self), 1)))
        
    def make_time_features(self, t, training=False, training_step=0):
        if isinstance(t, torch.Tensor):
            t = t.item()
            
        if self.cfg.normalize_timestamp:
            self.timestamp = t / self.max_timestamp
            offset_width = (1/self.max_frames)*0.1
        else:
            self.timestamp = t
            offset_width = 0.01
            
        if self.cfg.rescale_t:
            self.timestamp *= self.cfg.rescale_value
            offset_width *= self.factor_t_value
            
        if self.cfg.offset_t:
            self.timestamp += self.cfg.offset_value
            
        if self.cfg.random_noise and training:
            noise_weight = offset_width * (1 - (training_step/self.max_steps))
            self.timestamp += noise_weight*np.random.randn()
            
        return self.timestamp - self.time_center
        
    def set_timestep(self, t, training=False, training_step=0):

        timestamp = self.make_time_features(t, training, training_step)
            
        pos_base = self.position[:, :3]
        pos_traj_params = self.position[:, 3:].reshape(
            (len(self), self.cfg.pos_traj_dim, 3, self.xyz_traj_base_dim)
        )
        pos_traj = self.pos_fit_model(
            pos_traj_params, 
            timestamp, 
            self.pos_traj_base_dim
        )
        self.position_flow = pos_base + pos_traj
        
        rot_base = self.rotation[:, :4]
        rot_traj_params = self.rotation[:, 4:].reshape(
            (len(self), self.cfg.rot_traj_dim, 4, self.rot_traj_base_dim)
        )
        rot_traj = self.rot_fit_model(
            rot_traj_params, 
            timestamp, 
            self.rot_traj_base_dim
        )
        self.rotation_flow = rot_base + rot_traj

    @property
    def get_rotation_flow(self):
        return self.rotation_activation(self.rotation_flow)

    @property
    def get_position_flow(self):
        return self.position_flow
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation[:, :4])
    
    @property
    def get_position(self):
        return self.position[:, :3]
