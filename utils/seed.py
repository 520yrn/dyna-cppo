# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 21:58:13 2025

@author: Ruining
"""

import random
import numpy as np
import torch

def set_seed(seed: int = 0):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed set to {seed}")
