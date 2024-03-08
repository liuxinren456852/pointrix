import os
import argparse

import torch
from typing import List
from pointrix.utils.config import load_config
from pointrix.engine.default_trainer import DefaultTrainer

from model import DeformGaussian
from dataformat import NerfiesReFormat

class Trainer(DefaultTrainer):

    def train_step(self, batch: List[dict]) -> None:
        """
        The training step for the model.

        Parameters
        ----------
        batch : dict
            The batch data.
        """
        render_dict = self.model(batch, step=self.global_step)
        render_results = self.renderer.render_batch(render_dict, batch)
        self.loss_dict = self.model.get_loss_dict(render_results, batch)
        self.loss_dict['loss'].backward()
        self.optimizer_dict = self.model.get_optimizer_dict(self.loss_dict,
                                                            render_results,
                                                            self.white_bg)
    
    @torch.no_grad()
    def validation(self):
        self.val_dataset_size = len(self.datapipeline.validation_dataset)
        for i in range(0, self.val_dataset_size):
            self.call_hook("before_val_iter")
            batch = self.datapipeline.next_val(i)
            render_dict = self.model(batch, step=6000)
            render_results = self.renderer.render_batch(render_dict, batch)
            self.metric_dict = self.model.get_metric_dict(render_results, batch)
            self.call_hook("after_val_iter")

def main(args, extras) -> None:
    
    cfg = load_config(args.config, cli_args=extras)

    cfg.trainer.model.name = "DeformGaussian"
    cfg.trainer.dataset.data_type = "NerfiesReFormat"
    cfg.trainer.dataset.data_path = "/NASdata/clz/data/mochi-high-five"
    cfg['trainer']['optimizer']['optimizer_1']['params']['deform'] = {}
    cfg['trainer']['optimizer']['optimizer_1']['params']['deform']['lr'] = 0.00016 * 5.0
    cfg.trainer.val_interval = 5000

    gaussian_trainer = Trainer(
        cfg.trainer,
        cfg.exp_dir,
    )
    gaussian_trainer.train_loop()    
    model_path = os.path.join(
        cfg.exp_dir, 
        "chkpnt" + str(gaussian_trainer.global_step) + ".pth"
    )
    gaussian_trainer.save_model(model_path)
    print("\nTraining complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default = None)
    args, extras = parser.parse_known_args()
    
    main(args, extras)



    


