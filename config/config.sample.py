from pathlib import Path

import torch

from modules.utils import Mode

SEED = 42
DATA_DIR = Path("/project/data/NewArchive")
LOG_ROOT = Path("/project/results/logs")
MODE = Mode.ZERO_VS_ZERO_ONE
LOG_DIR = LOG_ROOT / f"tubles_{DATA_DIR.stem}_{MODE.name}"
WITH_TIMESTAMP = True
FROZEN = True
MAIN_METRIC = "accuracy01"  # "auc/_mean"

NUM_EPOCHS = 51
DEVICE = torch.device("cuda")
