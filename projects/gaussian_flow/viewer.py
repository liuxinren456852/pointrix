import numpy as np
import math
import torch

import os
import argparse
from einops import rearrange

from pointrix.utils.config import load_config
from trainer import GaussianFlow

# initialize the register
from data import dnerf_data
from gf_point import GaussianFlowPointCloud


def main(args, extras) -> None:
    
    cfg = load_config(args.config, cli_args=extras)
    gaussian_trainer = GaussianFlow(
        cfg.trainer,
        cfg.exp_dir,
    )
    ckpt_name = ""
    model_path = ""
    gaussian_trainer.load_model(path=model_path)
    
    gaussian_trainer.video_step(model_path, save_npz=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--smc_file", type=str, default = None)
    args, extras = parser.parse_known_args()
    
    main(args, extras)
