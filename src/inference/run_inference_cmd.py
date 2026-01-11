import os
import logging
import torch
import argparse
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
logger = logging.getLogger(__name__)

def load_safetensors_model_simple(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


if __name__ == "__main__":
    """
    Run inference in cmd
    """
    logging.basicConfig(filename='run_inference_cmd.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description="Train Llama-Guard for toxicity classification")
    parser.add_argument("--model-path", type=str, default="src/experiments/models/qwen_pretrain0.7B", help="Model to run locally")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=10)
    args = parser.parse_args()
    logger.info('Run parameters')
    logger.info(str(args))
    model_path = args.model_path
    logger.info('Load model: ' + model_path)
    model, tokenizer = load_safetensors_model_simple(model_path)
    model.eval()
    logger.info('Loaded model')

    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif 
    logger.info('Load model to GPU')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info('Loaded model to GPU')

    text = input("User:")
    
    with torch.no_grad():
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        print("==[TOKENIZED]==")
        outputs = model.generate(
            inputs.input_ids,
            max_length=128,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        print("==[GENERATED]==")
        generated_text = tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
    
    print("Model: ", generated_text)

    logger.info('Input processed. Generated text: ' + generated_text)