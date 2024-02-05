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
    '''
    Base class for all optimizers.
    '''
    def __init__(self, optimizer:Optimizer, **kwargs):
        self.optimizer = optimizer
        self.step = 1

    def update_model(self, loss: torch.Tensor) -> None:
        """
        update the model with the loss.

        Parameters
        ----------
        loss : torch.Tensor
            The loss tensor.
        """
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict(self) -> dict:
        """
        A wrapper of ``Optimizer.state_dict``.
        """
        state_dict = self.optimizer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """
        A wrapper of ``Optimizer.load_state_dict``.
        """
        # load state_dict of optimizer
        self.optimizer.load_state_dict(state_dict)

    def get_lr(self)-> Dict[str, List[float]]:
        """
        Get learning rate of the optimizer.

        Returns
        -------
        Dict[str, List[float]]
            The learning rate of the optimizer.
        """
        res = {}

        res['lr'] = [group['lr'] for group in self.optimizer.param_groups]
        return res

    def get_momentum(self) -> Dict[str, List[float]]:
        """
        Get momentum of the optimizer.

        Returns
        -------
        Dict[str, List[float]]
            The momentum of the optimizer.
        """
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
        """
        Get the parameter groups of the optimizer.

        Returns
        -------
        List[dict]
            The parameter groups of the optimizer.
        """
        return self.optimizer.param_groups
