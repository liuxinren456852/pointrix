import torch
from dataclasses import dataclass
from pointrix.utils.losses import l1_loss
from pointrix.point_cloud import parse_point_cloud
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY

@MODEL_REGISTRY.register()
class GaussianFlow(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        lambda_param_l1: float = 0.0
        lambda_knn: float = 0.0
    
    cfg: Config
    
    def setup(self, datapipline, device="cuda"):
        self.point_cloud = parse_point_cloud(
            self.cfg.point_cloud,
            datapipline
        ).to(device)
        self.point_cloud.set_prefix_name("point_cloud")
        self.device = device
        
        self.datapipline = datapipline
        self.global_step = 0
        # Set up flow paramters
        train_dataset = self.datapipline.training_dataset
        max_timestamp = train_dataset.camera_list[0].max_timestamp
        self.max_timestamp = max_timestamp
        self.point_cloud.max_timestamp = self.max_timestamp
        
    def get_gaussian(self):
        atributes_dict = {
            "opacity": self.point_cloud.get_opacity,
            "scaling": self.point_cloud.get_scaling,
        }
        return atributes_dict
    
    def get_flow(self):
        atributes_dict = {
            "position": self.point_cloud.get_position_flow,
            "rotation": self.point_cloud.get_rotation_flow,
            "shs": self.point_cloud.get_shs_flow,
        }
        return atributes_dict
    
    def forward(self, batch=None) -> dict:
        self.point_cloud.set_timestep(
            t=batch["camera"].timestamp,
            training=True,
            training_step=self.global_step,
        )
        return self.get_flow()
    
    def params_l1_regulizer(self):
        # random_choice = torch.sample (
        #     0, len(self.point_cloud), (10000, )
        # )
        # pos = self.point_cloud.position[:, 3:]
        # rot = self.point_cloud.rotation[:, 4:]
        pos = self.point_cloud.pos_params
        rot = self.point_cloud.rot_params
        pos_abs = torch.abs(pos)
        # pos_norm = pos_abs / pos_abs.max(dim=1, keepdim=True)[0]
        
        rot_abs = torch.abs(rot)
        # rot_norm = rot_abs / rot_abs.max(dim=1, keepdim=True)[0]
        
        loss_l1 = pos_abs.mean() + rot_abs.mean()
        # loss_norm = pos_norm.mean() + rot_norm.mean()
        
        return loss_l1 
    
    def get_loss_dict(self, render_results, batch) -> dict:

        gt_images = torch.stack(
            [batch[i]["image"].to(self.device) for i in range(len(batch))],
            dim=0
        )
        L1_loss = l1_loss(render_results['images'], gt_images)
        loss = L1_loss
        loss_dict = {
            "loss": loss,
            "L1_loss": L1_loss
        }
        
        if self.cfg.lambda_param_l1 > 0:
            param_l1 = self.params_l1_regulizer()
            loss_dict.update({
                "pl1_loss": param_l1,
            })
            loss += self.cfg.lambda_param_l1 * param_l1
            
        if self.cfg.lambda_knn > 0:
            if self.global_step == self.after_densifi_step:
                self.point_cloud.gen_knn()
                
            if self.global_step > self.after_densifi_step:
                knn_loss = self.point_cloud.knn_loss()
                loss_dict.update({
                    "knn_loss": knn_loss,
                })
                loss += self.cfg.lambda_knn * knn_loss
        
        return loss_dict
    