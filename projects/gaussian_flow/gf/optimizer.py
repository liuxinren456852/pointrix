import torch

from dataclasses import dataclass
from pointrix.utils.base import BaseObject
from pointrix.utils.config import C
from pointrix.optimizer.optimizer import OPTIMIZER_REGISTRY
from pointrix.optimizer.gs_optimizer import GaussianSplattingOptimizer
from pointrix.utils.gaussian_points.gaussian_utils import (
    inverse_sigmoid,
    build_rotation,
)

@OPTIMIZER_REGISTRY.register()
class FlowOptimizer(GaussianSplattingOptimizer):
    @dataclass
    class Config(GaussianSplattingOptimizer.Config):
        pass

    cfg: Config

    def new_pos_scale(self, mask):
        scaling = self.point_cloud.get_scaling
        position = self.point_cloud.position[:, :3]
        pos_traj = self.point_cloud.position[:, 3:]
        rotation = self.point_cloud.rotation[:, :4]
        split_num = self.split_num
        
        stds = scaling[mask].repeat(split_num, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        # TODO: make new rots depend on timestamp
        rots = build_rotation(
            rotation[mask]
        ).repeat(split_num, 1, 1)
        new_pos_base = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        ) + (
            position[mask].repeat(split_num, 1)
        )
        new_pos_traj = (
            pos_traj[mask].repeat(split_num, 1)
        )
        new_pos = torch.cat([new_pos_base, new_pos_traj], dim=-1)
        new_scaling = self.point_cloud.scaling_inverse_activation(
            scaling[mask].repeat(split_num, 1) / (0.8*split_num)
        )
        return new_pos, new_scaling
