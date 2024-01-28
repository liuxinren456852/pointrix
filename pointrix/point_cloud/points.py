

import torch
from torch import nn, Tensor
from dataclasses import dataclass, field
from pointrix.utils.base import BaseModule
from pointrix.utils.registry import Registry

from .utils import (
    unwarp_name,
    points_init,
)

POINTSCLOUD_REGISTRY = Registry("POINTSCLOUD", modules=["pointrix.point_cloud"])
POINTSCLOUD_REGISTRY.__doc__ = ""

@POINTSCLOUD_REGISTRY.register()
class PointCloud(BaseModule):
    @dataclass
    class Config:
        point_cloud_type: str = ""
        initializer: dict = field(default_factory=dict)
        trainable: bool = True
        unwarp_prefix: str = "point_cloud"
    
    cfg: Config
    
    def setup(self, point_cloud=None):
        self.atributes = []
        position, features = points_init(self.cfg.initializer, point_cloud)
        self.register_buffer('position', position)
        self.register_buffer('features', features)
        self.atributes.append({
            'name': 'position',
            'trainable': self.cfg.trainable,
        })
        self.atributes.append({
            'name': 'features',
            'trainable': self.cfg.trainable,
        })
        
        if self.cfg.trainable:
            self.position = nn.Parameter(
                position.contiguous().requires_grad_(True)
            )
            self.features = nn.Parameter(
                features.contiguous().requires_grad_(True)
            )
        self.prefix_name = self.cfg.unwarp_prefix + "."
            
    def unwarp(self, name):
        return unwarp_name(name, self.prefix_name)
            
    def get_params(self):
        params = {}
        for arr in self.atributes:
            name = arr['name']
            params.update({name: getattr(self, name)})
        return params
    
    def set_params(self, params):
        for name, value in params.items():
            setattr(self, name, value)
    
    def register_atribute(self, name, value, trainable=True):
        self.register_buffer(name, value)
        if self.cfg.trainable and trainable:
            setattr(
                self, 
                name, 
                nn.Parameter(
                    value.contiguous().requires_grad_(True)
                )
            )
        self.atributes.append({
            'name': name,
            'trainable': trainable,
        })
            
    def __len__(self):
        return len(self.position)
    
    def get_all_atributes(self):
        return self.atributes
    
    def select_atributes(self, mask):
        selected_atributes = {}
        for atribute in self.atributes:
            name = atribute['name']
            value = getattr(self, name)
            selected_atributes[name] = value[mask]
        return selected_atributes
    
    def replace(self, new_atributes, optimizer=None):
        if optimizer is not None:
            replace_tensor = self.replace_optimizer(
                new_atributes, 
                optimizer
            )
            for key, value in replace_tensor.items():
                setattr(self, key, value)
        else:
            for key, value in new_atributes.items():
                name = key
                value = getattr(self, name)
                replace_atribute = nn.Parameter(
                    value.contiguous().requires_grad_(True)
                )
                setattr(self, key, replace_atribute)
    
    def extand_points(self, new_atributes, optimizer=None):
        if optimizer is not None:
            extended_tensor = self.extend_optimizer(
                new_atributes, 
                optimizer
            )
            for key, value in extended_tensor.items():
                setattr(self, key, value)
        else:
            for atribute in self.atributes:
                name = atribute['name']
                value = getattr(self, name)
                extend_atribute = nn.Parameter(
                    torch.cat((
                        value, 
                        new_atributes['name']
                    ), dim=0).contiguous().requires_grad_(True)
                )
                setattr(self, key, extend_atribute)
    
    def remove_points(self, mask, optimizer=None):
        if optimizer is not None:
            prune_tensor = self.prune_optimizer(
                mask, 
                optimizer
            )
            for key, value in prune_tensor.items():
                setattr(self, key, value)
        else:
            for atribute in self.atributes:
                name = atribute['name']
                prune_value = nn.Parameter(
                    getattr(
                        self, name
                    )[mask].contiguous().requires_grad_(True)
                )
                setattr(self, key, prune_value)
    
    def prune_optimizer(self, mask, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if self.prefix_name in group["name"]:
                unwarp_ground = self.unwarp(group["name"])
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].contiguous().requires_grad_(True))
                    )
                    optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[unwarp_ground] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][mask].contiguous().requires_grad_(True)
                    )
                    optimizable_tensors[unwarp_ground] = group["params"][0]
        return optimizable_tensors
    
    def extend_optimizer(self, new_atributes, optimizer):
        new_tensors = {}
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            if self.prefix_name in group["name"]:
                unwarp_ground = self.unwarp(group["name"])
                extension_tensor = new_atributes[unwarp_ground]
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((
                        stored_state["exp_avg"], 
                        torch.zeros_like(extension_tensor)
                    ), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((
                        stored_state["exp_avg_sq"], 
                        torch.zeros_like(extension_tensor)
                    ), dim=0)

                    del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((
                            group["params"][0], 
                            extension_tensor
                        ), dim=0).contiguous().requires_grad_(True)
                    )
                    optimizer.state[group['params'][0]] = stored_state

                    new_tensors[unwarp_ground] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((
                            group["params"][0], 
                            extension_tensor
                        ), dim=0).contiguous().requires_grad_(True)
                    )
                    new_tensors[unwarp_ground] = group["params"][0]

        return new_tensors
    
    def replace_optimizer(self, new_atributes, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            for key, replace_tensor in new_atributes.items():
                if group["name"] == self.prefix_name + key:
                    stored_state = optimizer.state.get(group['params'][0], None)

                    stored_state["exp_avg"] = torch.zeros_like(replace_tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(replace_tensor)

                    del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        replace_tensor.contiguous().requires_grad_(True)
                    )
                    optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[self.unwarp(group["name"])] = group["params"][0]

        return optimizable_tensors