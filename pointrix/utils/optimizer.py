import numpy as np

import torch
import torch.nn as nn
# from torch.optim import lr_scheduler

from pointrix.utils import lr_scheduler

def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError

def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []

def parse_optimizer(config, model):
    if hasattr(config, "params"):
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
    else:
        params = model.parameters()
        
    if config.name in ["FusedAdam"]:
        import apex

        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler(config):
    if hasattr(config, "name"):
        scheduler = get_scheduler(config.name)
    
    max_steps = config.max_steps
    params = [
        {
            "name": name, 
            "init": values["init"], 
            "final": values["final"], 
            "max_steps": max_steps, 
        }
        for name, values in config.params.items()
    ]
    scheduler_funcs = {}
    for param in params:
        scheduler_funcs[param["name"]] = (
            scheduler(
                init=param["init"], 
                final=param["final"], 
                max_steps=param["max_steps"], 
            )
        )
    return scheduler_funcs