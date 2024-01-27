import torch
import torch.nn as nn
from pointrix.utils.config import parse_structured

from .base_optimizer import BaseOptimizer, OPTIMIZER_REGISTRY
from .gs_optimizer import GaussianSplattingOptimizer


__all__ = ["BaseOptimizer", "GaussianSplattingOptimizer"]


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

def parse_optimizer(config, model, **kwargs):
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

    optimizer_type = config.type
    optimizer = OPTIMIZER_REGISTRY.get(optimizer_type)
    return optimizer(optim, model, config.structure, **kwargs)