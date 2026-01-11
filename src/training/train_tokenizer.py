import argparse
import os
import yaml
import numpy
import torch
import random
from nip import load, wrap_module
from dotenv import load_dotenv
from .utils import set_random_seed

from .models.hf_bpe_tokenizer import HFBPETokenizerTrainer

load_dotenv()

def train_tokenizer_from_config(config_path, config):
    """
    Run main loop from config
    """
    set_random_seed(config["seed"])
    tokenizer_trainer = config["trainer"]
    tokenizer_trainer.train()
    tokenizer_trainer.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tokenizer")
    parser.add_argument("--config", type=str, default="src/configs/train_bpe.nip",
                        help="Path to config file (default: configs/train.nip")
    
    args = parser.parse_args()
    config = load(args.config)
    train_tokenizer_from_config(args.config, config)