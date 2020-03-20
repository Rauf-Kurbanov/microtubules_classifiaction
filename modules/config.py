from pathlib import Path

from modules.utils import Mode
import torch

SEED = 42
DATA_DIR = Path("/project/data/NewArchive")
# DATA_DIR = Path("/data/OldArchive")
LOG_ROOT = Path("/project/results/logs")
MODE = Mode.ZERO_ONE_VS_ONE
LOG_DIR = LOG_ROOT / f"tubles_{DATA_DIR.stem}_{MODE.name}"
WITH_TIMESTAMP = True

NUM_EPOCHS = 15
DEVICE = torch.device("cuda")
