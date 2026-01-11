import random
import numpy as np
import torch
import pandas as pd
from transformers import set_seed

def set_random_seed(seed=42):
    """
    Fix random seed for everything
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)