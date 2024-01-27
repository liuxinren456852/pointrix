from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer

from pointrix.utils.registry import Registry

OPTIMIZER_REGISTRY = Registry("OPTIMIZER", modules=["pointrix.optimizer"])
OPTIMIZER_REGISTRY.__doc__ = ""


@OPTIMIZER_REGISTRY.register()
class BaseOptimizer:
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer

        self.step = 1

    def update_model(self, loss: torch.Tensor) -> None:
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``."""
        state_dict = self.optimizer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:

        # load state_dict of optimizer
        self.optimizer.load_state_dict(state_dict)

    def get_lr(self):
        res = {}

        res['lr'] = [group['lr'] for group in self.optimizer.param_groups]

        return res

    def get_momentum(self) -> Dict[str, List[float]]:
        momentum = []
        for group in self.optimizer.param_groups:
            # Get momentum of SGD.
            if 'momentum' in group.keys():
                momentum.append(group['momentum'])
            # Get momentum of Adam.
            elif 'betas' in group.keys():
                momentum.append(group['betas'][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)
    
    @property
    def param_groups(self) -> List[dict]:
        """A wrapper of ``Optimizer.param_groups``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.param_groups
