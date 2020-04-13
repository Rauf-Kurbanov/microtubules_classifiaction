from pathlib import Path

import torch

from modules.utils import Mode

SEED = 42
PROJECT_ROOT = Path("/project")
DATA_DIR = PROJECT_ROOT / "data" / "OldArchive"
LOG_ROOT = PROJECT_ROOT / "results" / "embed"
MODE = Mode.ZERO_VS_ZERO_ONE_VS_ONE
WITH_TIMESTAMP = True
FROZEN = False
# MAIN_METRIC = "accuracy01"  # "auc/_mean"

NUM_EPOCHS = 20
DEVICE = torch.device("cuda")
N_WORKERS = 4
BATCH_SIZE = 64
WITH_AUGS = True
DEBUG = True
