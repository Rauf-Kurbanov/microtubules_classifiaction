from pathlib import Path

import torch

from modules.data import get_transforms, get_frozen_transforms
from modules.utils import Mode

SEED = 42
PROJECT_ROOT = Path("/Users/raufkurbanov/Programs/microtubules_classifiaction")
DATA_DIR = PROJECT_ROOT / "data" / "NewArchive"
LOG_ROOT = PROJECT_ROOT / "results" / "logs"
MODE = Mode.ZERO_VS_ZERO_ONE_VS_ONE
WITH_TIMESTAMP = True
FROZEN = True
MAIN_METRIC = "accuracy01"  # "auc/_mean"

NUM_EPOCHS = 1
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
TRANSFORMS = get_frozen_transforms()
