from pathlib import Path

import torch

from modules.utils import Mode

SEED = 42
PROJECT_ROOT = Path("/project")
DATA_DIR = PROJECT_ROOT / "data" / "OldArchive"
LOG_ROOT = PROJECT_ROOT / "results" / "logs"
MODE = Mode.ZERO_VS_ZERO_ONE_VS_ONE
WITH_TIMESTAMP = True
FROZEN = True
MAIN_METRIC = "accuracy01"  # "auc/_mean"

NUM_EPOCHS = 1
DEVICE = torch.device("cuda")
N_WORKERS = 4
BATCH_SIZE = 64
WITH_AUGS = True
# SIAMESE_CKPT = PROJECT_ROOT / "results" / "embed" / "OldArchive_ZERO_VS_ZERO_ONE_VS_ONE_FROZEN_Apr06_19-53-16" / "checkpoints" / "best.pth"
# SIAMESE_CKPT = None
SIAMESE_CKPT = PROJECT_ROOT / "results" / "embed" / "OldArchive_ZERO_VS_ZERO_ONE_VS_ONE__Apr06_20-49-06" / "checkpoints" / "best.pth"
DEBUG = True
