import functools
import logging
import numpy as np
from collections.abc import Mapping
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


def force_full_init(old_func: Callable) -> Any:
    """Those methods decorated by ``force_full_init`` will be forced to call
    ``full_init`` if the instance has not been fully initiated.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``full_init`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `full_init` method.
        if not hasattr(obj, 'full_init'):
            raise AttributeError(f'{type(obj)} does not have full_init '
                                 'method.')
        # If instance does not have `_fully_initialized` attribute or
        # `_fully_initialized` is False, call `full_init` and set
        # `_fully_initialized` to True
        if not getattr(obj, '_fully_initialized', False):
            print(
                f'Attribute `_fully_initialized` is not defined in '
                f'{type(obj)} or `type(obj)._fully_initialized is '
                'False, `full_init` will be called and '
                f'{type(obj)}._fully_initialized will be set to True',
                logger='current',
                level=logging.WARNING)
            obj.full_init()  # type: ignore
            obj._fully_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper
