
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import numpy as np

from pointrix.base_model.base import BaseObject

def unwarp_name(name):
    return name.replace("points_cloud.", "")

def get_random_points(p_size, radius):
    pos = np.random.random((p_size, 3)) * 2 * radius - radius
    pos = torch.from_numpy(pos).float()
    return pos

def get_random_feauture(p_size, feat_dim):
    feart = np.random.random((p_size, feat_dim)) / 255.0
    feart = torch.from_numpy(feart).float()
    return feart

def points_init(init_cfg):
    p_size = init_cfg.p_size
    init_type = init_cfg.init_type
    print("Number of points at initialisation : ", p_size)
    
    if init_type == 'random':
        pos = get_random_points(p_size, init_cfg.radius)
        features = get_random_feauture(p_size, init_cfg.feat_dim)

    return pos, features

class PointsCloud(BaseObject):
    @dataclass
    class Config:
        initializer: dict = field(default_factory=dict)
        trainable: bool = True
    
    cfg: Config
    
    def setup(self):
        self.atributes = []
        position, features = points_init(self.cfg.initializer)
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
            self.position = nn.Parameter(position.requires_grad_(True))
            self.features = nn.Parameter(features.requires_grad_(True))
    
    def register_atribute(self, name, value, trainable=True):
        self.register_buffer(name, value)
        if self.cfg.trainable and trainable:
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))
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
                    value.requires_grad_(True)
                )
                setattr(self, key, replace_atribute)
    
    def densify(self, new_atributes, optimizer=None):
        if optimizer is not None:
            extend_tensor = self.extend_optimizer(
                new_atributes, 
                optimizer
            )
            for key, value in extend_tensor.items():
                setattr(self, key, value)
        else:
            for atribute in self.atributes:
                name = atribute['name']
                value = getattr(self, name)
                extend_atribute = nn.Parameter(
                    torch.cat((
                        value, 
                        new_atributes['name']
                    ), dim=0).requires_grad_(True)
                )
                setattr(self, key, extend_atribute)
    
    def prune(self, mask, optimizer=None):
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
                    )[mask].requires_grad_(True)
                )
                setattr(self, key, prune_value)
    
    def prune_optimizer(self, mask, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            unwarp_ground = unwarp_name(group["name"])
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[unwarp_ground] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[unwarp_ground] = group["params"][0]
        return optimizable_tensors
    
    def extend_optimizer(self, new_atributes, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            unwarp_ground = unwarp_name(group["name"])
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
                    ), dim=0).requires_grad_(True)
                )
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[unwarp_ground] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((
                        group["params"][0], 
                        extension_tensor
                    ), dim=0).requires_grad_(True)
                )
                optimizable_tensors[unwarp_ground] = group["params"][0]

        return optimizable_tensors
    
    def replace_optimizer(self, new_atributes, optimizer):
        optimizable_tensors = {}
        for key, replace_tensor in new_atributes.items():
            for group in optimizer.param_groups:
                if group["name"] == "points_cloud."+key:
                    break
            stored_state = optimizer.state.get(group['params'][0], None)

            stored_state["exp_avg"] = replace_tensor
            stored_state["exp_avg_sq"] = replace_tensor

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(
                replace_tensor.requires_grad_(True)
            )
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[unwarp_name(group["name"])] = group["params"][0]

        return optimizable_tensors