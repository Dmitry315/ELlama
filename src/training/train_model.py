import argparse
import os
import yaml
import numpy
import torch
import random
from nip import load, wrap_module
from dotenv import load_dotenv
from utils import set_random_seed

from .models.hf_model_trainer import HFQwenTrainer
from .datasets_loading.pretrain_data import get_pretrain_data

load_dotenv()

def train_model_from_config(config_path, config):
    set_random_seed(config["seed"])
    model_trainer = config["trainer"]
    model_trainer.train()
    model_trainer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama-Guard for toxicity classification")
    parser.add_argument("--config", type=str, default="src/configs/train_bpe.nip",
                        help="Path to config file (default: configs/train.nip")
    
    args = parser.parse_args()

    config = load(args.config)
    train_model_from_config(args.config, config)