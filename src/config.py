"""
Kourosh - Configuration
Central settings: paths, reproducibility, device, hyperparameters.
"""
from pathlib import Path
import torch

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# ---------- Reproducibility ----------
SEED = 42

# ---------- Device ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Model hyperparameters ----------
VOCAB_SIZE = 50000
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
MAX_SEQ_LEN = 256
DROPOUT = 0.1

# ---------- Training ----------
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
GRAD_CLIP = 1.0