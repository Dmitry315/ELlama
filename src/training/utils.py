import random
import numpy as np
import torch
import pandas as pd
from transformers import set_seed
from trl import set_seed as trl_set_seed

def set_random_seed(seed=42):
    """
    Фиксирует random seed для всех библиотек
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # если используете multi-GPU
    
    # PyTorch дополнительные настройки для детерминизма
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Transformers (Hugging Face)
    set_seed(seed)
    
    # TRL (Transformer Reinforcement Learning)
    trl_set_seed(seed)
    
    # Pandas (хотя обычно не требует установки seed)
    # Но можно установить для воспроизводимости операций с random
    print(f"All random seeds set to: {seed}")