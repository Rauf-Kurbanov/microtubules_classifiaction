from pathlib import Path

import torch

from modules.data import get_transforms, get_frozen_transforms
from modules.utils import Mode

SEED = 42
PROJECT_ROOT = Path("/project")
DATA_DIR = PROJECT_ROOT / "data" / "OldArchive"
LOG_ROOT = PROJECT_ROOT / "results" / "embed"
MODE = Mode.ZERO_VS_ZERO_ONE_VS_ONE
WITH_TIMESTAMP = True
FROZEN = True
# MAIN_METRIC = "accuracy01"  # "auc/_mean"

NUM_EPOCHS = 1
DEVICE = torch.device("cuda")
BATCH_SIZE = 64
TRANSFORMS = get_transforms()
