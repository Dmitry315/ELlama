import argparse
import logging
import torch.distributed._shard.checkpoint as dist_cp
from accelerate import load_checkpoint_and_dispatch
from transformers import Qwen2ForCausalLM, PreTrainedTokenizerFast, Qwen2Config
logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(filename='fsdp_checkpoint_to_safetensors.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to safetensors")
    parser.add_argument("--fsdp-checkpoint", type=str)
    parser.add_argument("--output-model", type=str)
    parser.add_argument("--tokenizer", type=str)

    args = parser.parse_args()
    logger.info('Run parameters')
    logger.info(str(args))

    logger.info('Started laoding tokenizer')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    logger.info('Loaded tokenizer')
    
    logger.info('Started laoding Model')
    qwen_config = Qwen2Config(
        vocab_size=150000,
        hidden_size= 1024,
        head_dim= 128,
        intermediate_size= 3072,
        num_hidden_layers= 28,
        max_window_layers= 28,
        num_attention_heads= 16,
        num_key_value_heads= 8,
        max_position_embeddings= 2048,
        attention_dropout= 0.0,
        hidden_act= "silu",
        attention_bias= False
    )
    model = Qwen2ForCausalLM(config=qwen_config)
    logger.info('Loaded model')

    state_dict = {
            "model": model.state_dict()
        }

    logger.info('Start loading state dict')
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader= dist_cp.FileSystemReader(args.fsdp_checkpoint),
        no_dist=True,
    )
    logger.info('Dict loaded')
    logger.info('Start saving model to "' + args.output_model + '".')
    model.save_pretrained(args.output_model)
    tokenizer.save_pretrained(args.output_model)
    logger.info('Model is saved')

if __name__ == '__main__':
    main()