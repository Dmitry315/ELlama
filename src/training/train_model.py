import argparse
import os
import logging
import yaml
import numpy
import torch
import random
import torch.distributed as dist
from nip import load, wrap_module
from dotenv import load_dotenv
from utils import set_random_seed

from models.hf_model_trainer import HFQwenTrainer
from datasets_loading.pretrain_data import get_pretrain_data
logger = logging.getLogger(__name__)

load_dotenv()

def train_model_from_config(config_path, config):
    current_rank = dist.get_rank()
    logging.basicConfig(filename=f'run_model_train_rank_{current_rank}.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    logger.info("load_config from ", config_path)
    
    set_random_seed(config["seed"])
    model_trainer = config["trainer"]
    logger.info("Start train")
    model_trainer.train()
    logger.info("Start saving model")
    model_trainer.save()
    logger.info("Saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama-Guard for toxicity classification")
    parser.add_argument("--config", type=str, default="src/configs/train_bpe.nip",
                        help="Path to config file (default: configs/train.nip")
    
    args = parser.parse_args()

    config = load(args.config)
    train_model_from_config(args.config, config)