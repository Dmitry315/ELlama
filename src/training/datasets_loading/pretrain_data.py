from nip import nip
import json

from datasets import load_dataset, Dataset

def load_text_jsonl_data(file_path):
    """
    Load data from a JSONL file
    """
    texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                texts.append(data['text'])
    
    return Dataset.from_dict({
        'text': texts
    })

@nip
def get_pretrain_data(data_path, *args, **kwargs):
    """
    Load jsonl file

    can be used in nip-config
    """
    print("Load data")
    dataset = load_text_jsonl_data(data_path)
    print("Data loaded")
    # Format datasets
    return dataset

@nip
def get_stream_pretrain_data(data_path, *args, **kwargs):
    """
    Load jsonl file with streaming

    can be used in nip-config
    """
    print("Load data")
    dataset = load_dataset('json', data_files=data_path, streaming=True)
    print("Data loaded")
    return dataset