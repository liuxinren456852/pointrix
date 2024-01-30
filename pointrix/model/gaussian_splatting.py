from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from tqdm import tqdm
from pointrix.engine.default_trainer import DefaultTrainer
from pointrix.exporter.novel_view import test_view_render, novel_view_render

class GaussianSplatting(DefaultTrainer):
    @dataclass
    class Config(DefaultTrainer.Config):
        max_sh_degree: int = 3
        # Train cfg
        lambda_dssim: float = 0.2        
        densification: dict = field(default_factory=dict)

    cfg: Config

    def train_step(self, batch) -> Dict:
        render_dict = self.point_cloud(batch)
        render_results = self.renderer.render_batch(render_dict, batch)
        self.loss_dict = self.point_cloud.get_loss_dict(render_results, batch)
        self.optimizer_dict = self.point_cloud.get_optimizer_dict(self.loss_dict, 
                                                                  render_results, self.white_bg)
        
    @torch.no_grad()
    def validation(self):
        self.val_dataset_size = len(self.datapipline.validation_dataset)
        progress_bar = tqdm(
                        range(0, self.val_dataset_size),
                        desc="Validation progress",
                        leave=False,
                    )
        for i in range(0, self.val_dataset_size):
            self.call_hook("before_val_iter")
            batch = self.datapipline.next_val()
            render_dict = self.point_cloud(batch)
            render_results = self.renderer.render_batch(render_dict, batch)
            self.metric_dict = self.point_cloud.get_metric_dict(render_results, batch)
            self.call_hook("after_val_iter")
            progress_bar.update(1)
        progress_bar.close()
        self.call_hook("after_val")


    @torch.no_grad()
    def test(self):
        self.point_cloud.load_ply('/home/clz/code_remote/Pointrix/projects/gaussian_splatting/garden/30000.ply')
        self.point_cloud.to(self.device)
        test_view_render(self.point_cloud, self.renderer, self.datapipline, output_path=self.cfg.output_path)
        novel_view_render(self.point_cloud, self.renderer, self.datapipline, output_path=self.cfg.output_path)
        
    def saving(self):
        data_list = {
            "active_sh_degree": self.active_sh_degree,
            "point_cloud": self.point_cloud.state_dict(),
        }
        return data_list