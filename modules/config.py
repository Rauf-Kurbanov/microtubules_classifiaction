from pathlib import Path

from modules.utils import Mode
import torch

SEED = 42
# DATA_DIR = Path("/data/NewArchive")
DATA_DIR = Path("/data/OldArchive")
LOG_ROOT = Path("/data/logs")
MODE = Mode.ZERO_ONE_VS_ONE
NUM_EPOCHS = 15
DEVICE = torch.device("cuda")
