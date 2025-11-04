import os
import pytest
import subprocess
from transformers import Qwen2ForCausalLM
from data_process.prepare_fineweb import read_fine_web, save_txt
from convert.fsdp_checkpoint_to_safetensors import convert_fsdp_to_checkpoint
from inference.run_inference_cmd import load_safetensors_model_simple
from training.train_tokenizer import train_tokenizer_from_config
from training.models.hf_bpe_tokenizer import HFBPETokenizerTrainer


def test_download_fineweb():
    path = "tests/data/fineweb2_example/"
    os.makedirs(path, exist_ok=True)
    read_fine_web(save_path=path, dataset_name="dmitry315/fineweb2-modern-greece-sample-test")
    assert os.path.exists(path + "/train.jsonl")
    assert os.path.exists(path + "/val.jsonl")
    save_txt(read_path=path + "/train.jsonl", save_path=path + "/train.txt")
    assert os.path.exists(path + "/train.txt")

# def test_convert():
#     convert_fsdp_to_checkpoint("tests/data/fsdp_checkpoint/pytorch_model_fsdp_0/", "tests/data/final_model", "dmitry315/ELlama1-0.7b")
#     assert os.path.exists("tests/data/final_model/config.json")
#     assert os.path.exists("tests/data/final_model/tokenizer.json")

def load_model():
    model, tokenizer = load_safetensors_model_simple("dmitry315/ELlama1-0.7b")
    assert type(model) is Qwen2ForCausalLM
    assert tokenizer("Πιστεύω ότι το νόημα της ζωής βρίσκεται στο")["input_ids"] == [12801, 4211, 4097, 9278, 4118, 5321, 5170, 4132]

def test_train_tokenizer():
    path = "tests/data/fineweb2_example/"
    os.makedirs(path, exist_ok=True)
    read_fine_web(save_path=path, dataset_name="dmitry315/fineweb2-modern-greece-sample-test")
    save_txt(read_path=path + "/train.jsonl", save_path=path + "/train.txt")
    
    trainer_config = {
        "bpe_init_params": {"dropout": 0},
        "bpe_trainer_params": {
            "vocab_size": 10,
            "min_frequency": 1,
            "special_tokens": ["<|bos|>", "<|unk|>", "<|sep|>", "<|pad|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]
        },
        "train_corpus_files": [path + "/train.txt"],
        "save_path": "tests/bpe_tokenizer",
        "tokenizer_fast_params": {
            "padding_side": "left",
            "truncation_side": "left",
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "unk_token": "<|unk|>",
            "sep_token": "<|sep|>",
            "pad_token": "<|pad|>",
            "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]
        }
    }
    config = {
        "seed": 42,
        "trainer": HFBPETokenizerTrainer(**trainer_config)
    }
    
    train_tokenizer_from_config(config_path=None, config=config)

    assert os.path.exists("tests/bpe_tokenizer/tokenizer.json")
    assert os.path.exists("tests/bpe_tokenizer/tokenizer_config.json")
    assert os.path.exists("tests/bpe_tokenizer/special_tokens_map.json")