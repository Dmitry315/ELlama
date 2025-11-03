from nip import nip
import torch
import torch.distributed as dist
import logging

from trl import SFTTrainer, SFTConfig
from transformers import PreTrainedTokenizerFast, Qwen2ForCausalLM, Qwen2Config
logger = logging.getLogger(__name__)

@nip
class HFQwenTrainer:
    def __init__(self, 
                 qwen_params, 
                 tokenizer_path, 
                 trainer_config_params, 
                 train_dataset, 
                 val_dataset,
                 save_path, 
                 resume_from_checkpoint=False,
                 add_size_to_name=True,
                 tokenizer_truncation_side="right",
                 tokenizer_padding_side="left",
                 use_accelerate=True,
                 validation_callback_params=dict(),
                 *args, **kwargs
                 ):
        current_rank = dist.get_rank()
        logging.basicConfig(filename=f'qwen_trainer_{current_rank}.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        logger.info('Load tokenizer')
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer.padding_side = tokenizer_padding_side
        self.tokenizer.truncation_side = tokenizer_truncation_side

        logger.info(f"pad_token: {self.tokenizer.pad_token}")
        logger.info(f"pad_token_id: {self.tokenizer.pad_token_id}")
        logger.info(f"eos_token: {self.tokenizer.eos_token}")
        logger.info(f"eos_token_id: {self.tokenizer.eos_token_id}")

        logger.info('Load model')
        self.qwen_config = Qwen2Config(
            vocab_size= self.tokenizer.vocab_size,
            **qwen_params
        )
        self.model = Qwen2ForCausalLM(config=self.qwen_config)
        logger.info('Load config')
        self.train_config = SFTConfig(**trainer_config_params)
        logger.info(str(self.tokenizer))
        logger.info(str(self.train_dataset))
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_path = save_path
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("params:  {total_params}")
        end = ""
        if total_params >= 100_000_000:
            total_params /= 1_000_000_000
            end = "B"
        elif total_params >= 100_000:
            total_params //= 1_000_000
            end = "M"
        logger.info(f"Total parameters: {total_params:.1f}"+end)
        if add_size_to_name:
            self.save_path += f"{total_params:.1f}" + end

        self.save_path = self.save_path.replace(" ", "_")

        self.use_accelerate = use_accelerate

        callbacks = []
        logger.info('Load trainer')
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=self.train_config,
            callbacks=callbacks
        )

    def train(self):
        current_rank = dist.get_rank()
        logging.basicConfig(filename=f'qwen_trainer_{current_rank}.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger.info("Start train")
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        logger.info("Model trained")

    def save(self):
        current_rank = dist.get_rank()
        logging.basicConfig(filename=f'qwen_trainer_{current_rank}.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
        if self.use_accelerate:
            logger.info("Save fsdp model")
            self.trainer.accelerator.save_state(output_dir=self.save_path, safe_serialization=True)
        else:
            logger.info("Save safetensor model")
            self.model.save_pretrained(self.save_path)
            self.tokenizer.save_pretrained(self.save_path)
        logger.info("Model saved")