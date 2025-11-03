from nip import nip
import logging

from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers.normalizers import NFD, StripAccents
logger = logging.getLogger(__name__)

@nip
class HFBPETokenizerTrainer:
    def __init__(self, 
                 bpe_init_params, 
                 bpe_trainer_params, 
                 train_corpus_files, 
                 save_path,
                 tokenizer_fast_params,
                 *args, **kwargs):
        logging.basicConfig(filename='bpe_trainer.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger.info('Load tokenizer')
        self.tokenizer = Tokenizer(BPE(**bpe_init_params))
        logger.info('Load trainer')
        self.trainer = BpeTrainer(**bpe_trainer_params)
        logger.info('Load pretokenizer')
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        logger.info('Load normalizer')
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
        self.train_corpus_files = train_corpus_files
        self.save_path = save_path
        self.tokenizer_fast_params = tokenizer_fast_params

    def train(self):
        logging.basicConfig(filename='bpe_trainer.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger.info('Start training')
        self.tokenizer.train(files=self.train_corpus_files, trainer=self.trainer)
        logger.info('End training')

    def save(self):
        logging.basicConfig(filename='bpe_trainer.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger.info('Save tokenizer')
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer, **self.tokenizer_fast_params)
        hf_path = self.save_path.get("hub", None)
        local_path = self.save_path.get("local_path", None)
        if local_path is not None:
            logger.info('Save locally: ' + local_path)
            fast_tokenizer.save_pretrained(local_path)
        if hf_path is not None:
            logger.info('Save to hub: ' + hf_path)
            fast_tokenizer.push_to_hub(hf_path)
        logger.info('Tokenizer saved')
