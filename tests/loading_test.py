import os
import pytest
import subprocess
from transformers import Qwen2ForCausalLM
from data_process.prepare_fineweb import read_fine_web, save_txt
from convert.fsdp_checkpoint_to_safetensors import convert_fsdp_to_checkpoint
from inference.run_inference_cmd import load_safetensors_model_simple

@pytest.mark.slow
def test_download_fineweb():
    path = "tests/data/fineweb2_example/"
    os.makedirs(path, exist_ok=True)
    read_fine_web(save_path=path)
    assert os.path.exists(path + "/train.jsonl")
    assert os.path.exists(path + "/val.jsonl")
    save_txt(read_path=path + "/train.jsonl", save_path=path + "/train.txt")
    assert os.path.exists(path + "/train.txt")
    os.rmtree(path)

@pytest.mark.slow
def test_convert():
    convert_fsdp_to_checkpoint("tests/data/fsdp_checkpoint/pytorch_model_fsdp_0/", "tests/data/final_model", "dmitry315/ELlama1-0.7b")
    assert os.path.exists("tests/data/final_model/config.json")
    assert os.path.exists("tests/data/final_model/tokenizer.json")

@pytest.mark.slow
def load_model():
    model, tokenizer = load_safetensors_model_simple("dmitry315/ELlama1-0.7b")
    assert type(model) is Qwen2ForCausalLM
    assert tokenizer("Πιστεύω ότι το νόημα της ζωής βρίσκεται στο")["input_ids"] == [12801, 4211, 4097, 9278, 4118, 5321, 5170, 4132]