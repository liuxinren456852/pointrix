import os
import argparse

import sys
sys.path.append("../../")
from pointrix.utils.config import load_config
from pointrix.engine.default_trainer import DefaultTrainer

from model import DeformGaussian
from dataformat import NerfiesReFormat

def main(args, extras) -> None:
    
    cfg = load_config(args.config, cli_args=extras)

    cfg.trainer.model.name = "DeformGaussian"
    cfg.trainer.dataset.data_type = "NerfiesReFormat"
    cfg.trainer.dataset.data_path = "/home/clz/data/dnerf/cat"
    cfg['trainer']['optimizer']['optimizer_1']['params']['deform'] = {}
    cfg['trainer']['optimizer']['optimizer_1']['params']['deform']['lr'] = 0.00016 * 5.0
    cfg.trainer.val_interval = 5000

    gaussian_trainer = DefaultTrainer(
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



    


