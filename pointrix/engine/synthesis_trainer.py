import os
import random
from tqdm import tqdm
from typing import Any, Optional, Union, List
from dataclasses import dataclass, field

import torch
from torch import nn
from pathlib import Path
from pointrix.renderer import parse_renderer
from pointrix.dataset import parse_data_pipeline
from pointrix.utils.config import parse_structured
from pointrix.optimizer import parse_optimizer, parse_scheduler
from pointrix.model import parse_model
from pointrix.logger import parse_writer
from pointrix.hook import parse_hooks
from pointrix.exporter.novel_view import test_view_render, novel_view_render

from .default_trainer import DefaultTrainer
from torch.utils.tensorboard import SummaryWriter
import imageio

class Synthesis_Trainer(DefaultTrainer):
    """
    The default trainer class for training and testing the model.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.
    exp_dir : str
        The experiment directory.
    device : str, optional
        The device to use, by default "cuda".
    """
    def __init__(self, cfg,  exp_dir: Path, device: str = "cuda") -> None:
        super().__init__(cfg,exp_dir,device)


    def train_loop(self) -> None:
        """
        The training loop for the model.
        """
        loop_range = range(self.start_steps, self.cfg.max_steps+1)
        self.global_step = self.start_steps
        self.call_hook("before_train")
        for iteration in loop_range:
            self.call_hook("before_train_iter")
            radius_now_scale=1+self.cfg['dataset']['generate_cfg']['radius_increase_scale'] *self.global_step
            self.datapipeline.set_all_scale(radius_now_scale)
            batch = self.datapipeline.next_train(self.global_step)
            self.renderer.update_sh_degree(iteration)
            self.schedulers.step(self.global_step, self.optimizer)
            self.train_step(batch)
            self.optimizer.update_model(**self.optimizer_dict)
            self.call_hook("after_train_iter")
            self.global_step += 1
            if (iteration+1)%self.cfg.video_interval==0:
                self.video_inference(iteration+1)
            if (iteration+1) % self.cfg.val_interval == 0 or iteration == self.cfg.max_steps:
                self.call_hook("before_val")
                self.validation()
                self.call_hook("after_val")
        self.call_hook("after_train")
    
    def train_step(self, batch: List[dict]) -> None:
        """
        The training step for the model.

        Parameters
        ----------
        batch : dict
            The batch data.
        """
        render_dict = self.model(batch)
        render_results = self.renderer.render_batch(render_dict, batch)
        self.loss_dict = self.model.get_loss_dict(render_results, batch,render_dict=render_dict,global_step=self.global_step)
        self.loss_dict['loss'].backward()
        self.optimizer_dict = self.model.get_optimizer_dict(self.loss_dict,
                                                            render_results,
                                                            self.white_bg)

    
    def video_inference(self, iteration):
        self.datapipeline.validation_dataset.resample()
        self.val_dataset_size = len(self.datapipeline.validation_dataset)
        save_folder = os.path.join(
            self.exp_dir, "videos/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  # makedirs
            print('videos is in :', save_folder)
        torch.cuda.empty_cache()
        img_frames = []
        for i in range(0, self.val_dataset_size):
            # self.call_hook("before_val_iter")
            batch = self.datapipeline.next_val(i)
            render_dict = self.model(batch)
            render_results = self.renderer.render_batch(render_dict, batch)
            rgbs = render_results["rgb"]
            for index in range(rgbs.shape[0]):
                rgb = rgbs[index, :, :, :]
                image = torch.clamp(rgb, 0.0, 1.0)
                image = image.detach().cpu().permute(1, 2, 0).numpy()
                image = (image * 255).round().astype('uint8')
                img_frames.append(image)
                # self.call_hook("after_val_iter")
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(
            iteration)), img_frames, fps=30, quality=8)
        print("\n[ITER {}] Video Save Done!".format(iteration))
        torch.cuda.empty_cache()
