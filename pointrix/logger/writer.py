import socket
import torch
import os
from pathlib import Path
from torch import Tensor
from jaxtyping import Float
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.progress import (BarColumn, Progress, ProgressColumn,
                           Task, TaskProgressColumn, TextColumn,
                           TimeRemainingColumn)
from rich.text import Text

Logger = Console(width=120)


from pointrix.utils.registry import Registry

LOGGER_REGISTRY = Registry("LOGGER", modules=["pointrix.hook"])
LOGGER_REGISTRY.__doc__ = ""


class ItersPerSecColumn(ProgressColumn):
    """Renders the iterations per second for a progress bar."""

    def __init__(self, suffix="it/s") -> None:
        super().__init__()
        self.suffix = suffix

    def render(self, task: Task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:.2f} {self.suffix}", style="progress.data.speed")


def create_progress(description: str, suffix: Optional[str] = None):
    """Helper function to return a rich Progress object."""
    progress_list = [TextColumn(description), BarColumn(
    ), TaskProgressColumn(show_speed=True)]
    progress_list += [ItersPerSecColumn(suffix=suffix)] if suffix else []
    progress_list += [TimeRemainingColumn(
        elapsed_when_finished=True, compact=True)]
    progress = Progress(*progress_list)
    return progress


class Writer:
    """
    Base class for writers.

    Parameters
    ----------
    log_dir : Path
        The directory to save the logs.
    """

    def __init__(self, log_dir: Path):
        self.logfolder = log_dir

    @abstractmethod
    def write_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int):
        """
        Write a scalar value to the writer.

        Parameters
        ----------
        name : str
            The name of the scalar.
        scalar : Union[float, torch.Tensor]
            The scalar value.
        step : int
            The step of the scalar.
        """
        assert NotImplementedError

    @abstractmethod
    def write_image(self, name: str, image: Float[Tensor, "H W C"], step: int, caption: Union[str, None] = None):
        """
        Write an image to the writer.

        Parameters
        ----------
        name : str
            The name of the image.
        image : Float[Tensor, "H W C"]
            The image.
        step : int
            The step of the image.
        caption : Union[str, None], optional
            The caption of the image, by default None
        """
        assert NotImplementedError

    @abstractmethod
    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):
        """
        Write a config to the writer.

        Parameters
        ----------
        name : str
            The name of the config.
        config_dict : Dict[str, Any]
            The config.
        step : int
            The step of the config.
        """
        assert NotImplementedError

@LOGGER_REGISTRY.register()
class TensorboardWriter(Writer):
    """
    Tensorboard writer.

    Parameters
    ----------
    log_dir : Path
        The directory to save the logs.
    """

    def __init__(self, log_dir, **kwargs):
        self.writer = SummaryWriter(log_dir)

    def write_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int):
        """
        Write a scalar value to the writer.

        Parameters
        ----------
        name : str
            The name of the scalar.
        scalar : Union[float, torch.Tensor]
            The scalar value.
        step : int
            The step of the scalar.
        """
        self.writer.add_scalar(name, scalar, global_step=step)

    def write_image(self, name: str, image: Float[Tensor, "H W C"], step: int, caption: Union[str, None] = None):
        """
        Write an image to the writer.

        Parameters
        ----------
        name : str
            The name of the image.
        image : Float[Tensor, "H W C"]
            The image.
        step : int
            The step of the image.
        caption : Union[str, None], optional
            The caption of the image, by default None
        """
        self.writer.add_image(name, image, step)

    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):
        """
        Write a config to the writer.

        Parameters
        ----------
        name : str
            The name of the config.
        config_dict : Dict[str, Any]
            The config.
        step : int
            The step of the config.
        """
        self.writer.add_text("config", str(config_dict))

@LOGGER_REGISTRY.register()
class WandbWriter(Writer):
    def __init__(self, log_dir, experiment_name: str, project_name: str = "pointrix-project"):
        """
        Wandb writer.

        Parameters
        ----------
        log_dir : Path
            The directory to save the logs.
        experiment_name : str
            The name of the experiment.
        project_name : str, 
            The name of the project, by default "pointrix-project"
        """
        import wandb
        wandb.init(project=project_name,
                   name=experiment_name,
                   dir=log_dir,
                   reinit=True)

    def write_scalar(self, name: str, scalar: Union[float, torch.Tensor], step: int):
        """
        Write a scalar value to the writer.

        Parameters
        ----------
        name : str
            The name of the scalar.
        scalar : Union[float, torch.Tensor]
            The scalar value.
        step : int
            The step of the scalar.
        """
        import wandb
        wandb.log({name: scalar}, step=step)

    def write_image(self, name: str, image: Float[Tensor, "H W C"], step: int, caption=None):
        """
        Write an image to the writer.

        Parameters
        ----------
        name : str
            The name of the image.
        image : Float[Tensor, "H W C"]
            The image.
        step : int
            The step of the image.
        caption : Union[str, None], optional
            The caption of the image, by default None
        """
        import wandb
        wandb.log(
            {name: [wandb.Image(image, caption=name if caption == None else caption)]})

    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):
        """Function that writes out the config to wandb

        Args:
            config: config dictionary to write out
        """
        import wandb

        wandb.config.update(config_dict, allow_val_change=True)
