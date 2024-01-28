import os
import argparse

import sys
from pointrix.utils.config import load_config
from trainer import GaussianFlow

# initialize the register
from data import dnerf_data
from gf_point import GaussianFlowPointCloud

import taichi as ti
ti.init(arch=ti.cuda)

def main(args, extras) -> None:
    
    cfg = load_config(args.config, cli_args=extras)
    gaussian_trainer = GaussianFlow(
        cfg.trainer,
        cfg.exp_dir,
    )
    
    gaussian_trainer.train_loop()    
    model_path = os.path.join(
        cfg.exp_dir, 
        "chkpnt" + str(gaussian_trainer.global_step) + ".pth"
    )
    gaussian_trainer.save_model(path=model_path)
    
    gaussian_trainer.test()
    print("\nTraining complete.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default = None)
    args, extras = parser.parse_known_args()
    
    main(args, extras)
    