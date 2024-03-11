import os
import argparse

import sys
from pointrix.utils.config import load_config
from trainer import GaussianFlowTrainer

# initialize the register
import gf
from data import dnerf_data


import taichi as ti
ti.init(arch=ti.cuda)

def main(args, extras) -> None:
    
    cfg = load_config(args.config, cli_args=extras)
    gaussian_trainer = GaussianFlowTrainer(
        cfg.trainer,
        cfg.exp_dir,
    )
    
    gaussian_trainer.train_loop()    
    model_path = os.path.join(
        gaussian_trainer.cfg.output_path, 
        "{}.pth".format(gaussian_trainer.global_step-1)
    )
    gaussian_trainer.save_model(path=model_path)
    
    gaussian_trainer.test(model_path)
    print("\nTraining complete.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default = None)
    args, extras = parser.parse_known_args()
    
    main(args, extras)
    