import argparse
import os
import yaml
import logging
import numpy
import torch
import random
from nip import load, wrap_module
from dotenv import load_dotenv
import torch.distributed as dist
from utils import set_random_seed

from models.hf_bpe_tokenizer import HFBPETokenizerTrainer
logger = logging.getLogger(__name__)

load_dotenv()

def train_tokenizer_from_config(config_path, config):
    current_rank = dist.get_rank()
    logging.basicConfig(filename=f'run_tokenizer_train_rank_{current_rank}.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger.info("load_config from ", config_path)
    set_random_seed(config["seed"])
    logger.info("Start train")
    tokenizer_trainer = config["trainer"]
    tokenizer_trainer.train()
    logger.info("Start saving model")
    tokenizer_trainer.save()
    logger.info("Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama-Guard for toxicity classification")
    parser.add_argument("--config", type=str, default="src/configs/train_bpe.nip",
                        help="Path to config file (default: configs/train.nip")
    
    args = parser.parse_args()
    config = load(args.config)
    train_tokenizer_from_config(args.config, config)