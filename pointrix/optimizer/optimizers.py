import torch
import torch.nn as nn
from torch.optim import Optimizer
from .base_optimizer import BaseOptimizer


class Optimizers:
    """
    A wrapper for multiple optimizers.
    """
    def __init__(self, optimizer_dict: dict) -> None:
        """
        Parameters
        ----------
        optimizer_dict : dict
            The dictionary of the optimizers.
        """
        for key, value in optimizer_dict.items():
            assert isinstance(value, BaseOptimizer), (
                '`OptimWrapperDict` only accept BaseOptimizer instance, '
                f'but got {key}: {type(value)}')
        self.optimizer_dict = optimizer_dict
    
    def update_model(self, loss, **kwargs) -> None:
        """
        update the model with the loss.

        Parameters
        ----------
        loss : torch.Tensor
            The loss tensor.
        kwargs : dict
            The keyword arguments.
        """
        loss.backward()
        for name, optimizer in self.optimizer_dict.items():
            optimizer.update_model(loss, **kwargs)
    
    def state_dict(self) -> dict:
        """
        A wrapper of ``Optimizer.state_dict``.

        Returns
        -------
        dict
            The state dictionary of the optimizer.
        """
        state_dict = dict()
        for name, optimizer in self.optimizer_dict.items():
            state_dict[name] = optimizer.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        A wrapper of ``Optimizer.load_state_dict``.

        Parameters
        ----------
        state_dict : dict
            The state dictionary of the optimizer.
        """
        for name, _state_dict in state_dict.items():
            assert name in self.optimizer_dict, (
                f'Mismatched `state_dict`! cannot found {name} in '
                'OptimWrapperDict')
            self.optimizer_dict[name].load_state_dict(_state_dict)

    def __len__(self) -> int:
        """
        Get the number of the optimizers.

        Returns
        -------
        int
            The number of the optimizers.
        """
        return len(self.optimizer_dict)
    
    def __contains__(self, key: str) -> bool:
        """
        Check if the key is in the optimizer dictionary.

        Parameters
        ----------
        key : str
            The key to check.
        
        Returns
        -------
        bool
            Whether the key is in the optimizer dictionary.
        """
        return key in self.optimizer_dict
    
    @property
    def param_groups(self):
        """
        Get the parameter groups of the optimizers.

        Returns
        -------
        list
            The parameter groups of the optimizers.
        """
        param_groups = []
        for key, value in self.optimizer_dict.items():
            param_groups.extend(value.param_groups)
        return param_groups