import torch

from tqdm import tqdm
from typing import List
from dataclasses import dataclass
from pointrix.engine.default_trainer import DefaultTrainer
from novel_view import test_view_render, novel_view_render
class GaussianFlowTrainer(DefaultTrainer):
    @dataclass
    class Config(DefaultTrainer.Config):
        grad_clip_value: float = 0.0
    
    cfg: Config
    
    def before_train(self):
        # Densification setup d
        self.after_densifi_step = self.cfg.densification.densify_stop_iter+1
        
    def train_step(self, batch: List[dict]) -> None:
        # update global_step in model
        self.model.global_step = self.global_step
        render_results = self.renderer.render_batch(self.model, batch)
        self.loss_dict = self.model.get_loss_dict(render_results, batch)
        
        self.loss_dict['loss'].backward()
        
        if self.cfg.grad_clip_value > 0:
            torch.nn.utils.clip_grad_value_(
                self.model.point_cloud.parameters(), 
                self.cfg.grad_clip_value
            )
        
        self.optimizer_dict = self.model.get_optimizer_dict(
            self.loss_dict,
            render_results,
            self.white_bg
        )
        
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
            batch = self.datapipline.next_val(i)
            render_results = self.renderer.render_batch(self.model, batch)
            self.metric_dict = self.model.get_metric_dict(
                render_results, batch)
            self.call_hook("after_val_iter")
            progress_bar.update(1)
        progress_bar.close()
        self.call_hook("after_val")
        
    @torch.no_grad()
    def test(self, model_path) -> None:
        """
        The testing method for the model.
        """
        self.model.load_ply(model_path)
        self.model.to(self.device)
        self.renderer.active_sh_degree = 3
        
        def render_func(data):
            static_gaussian = self.model.get_gaussian()
            data.update(static_gaussian)
            self.model.point_cloud.set_timestep(
                t=data["camera"].timestamp,
            )
            flow_gaussian = self.model.get_flow()
            data.update(flow_gaussian)
            return self.renderer.render_iter(**data)
        
        test_view_render(
            render_func,
            self.datapipline,
            output_path=self.cfg.output_path
        )
        novel_view_render(
            render_func,
            self.datapipline, 
            output_path=self.cfg.output_path
        )
        
