from nip import nip
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers.normalizers import NFD, StripAccents

@nip
class HFBPETokenizerTrainer:
    """
    BPE trainer class

    Can be used in nip-config
    """

    def __init__(self, 
                 bpe_init_params, 
                 bpe_trainer_params, 
                 train_corpus_files, 
                 save_path,
                 tokenizer_fast_params,
                 *args, **kwargs):
        print('Load tokenizer')
        self.tokenizer = Tokenizer(BPE(**bpe_init_params))
        print('Load trainer')
        self.trainer = BpeTrainer(**bpe_trainer_params)
        print('Load pretokenizer')
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        print('Load normalizer')
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
        self.train_corpus_files = train_corpus_files
        self.save_path = save_path
        self.tokenizer_fast_params = tokenizer_fast_params

    def train(self):
        """
        Start HF training loop
        """
        print('Start training')
        self.tokenizer.train(files=self.train_corpus_files, trainer=self.trainer)
        print('End training')

    def save(self):
        """
        Save tokenizer locally or on HF
        """
        print('Save tokenizer')
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer, **self.tokenizer_fast_params)
        hf_path = self.save_path.get("hub", None)
        local_path = self.save_path.get("local_path", None)
        if local_path is not None:
            print('Save locally: ' + local_path)
            fast_tokenizer.save_pretrained(local_path)
        if hf_path is not None:
            print('Save to hub: ' + hf_path)
            fast_tokenizer.push_to_hub(hf_path)
        print('Tokenizer saved')
