"""Utility helpers for Kourosh."""
import random
import numpy as np
import torch
from src.config import SEED


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across random/numpy/torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)