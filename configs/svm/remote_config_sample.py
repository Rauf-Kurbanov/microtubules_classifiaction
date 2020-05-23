from pathlib import Path

import torch

from modules.utils import Mode

SEED = 42
PROJECT_ROOT = Path("/project")
DATASET_NAME = "FilteredCleanProcessed"
DATA_DIR = PROJECT_ROOT / "data" / DATASET_NAME
LOG_ROOT = PROJECT_ROOT / "results" / "embed"
MODE = Mode.ZERO_VS_ZERO_ONE_VS_ONE
WITH_TIMESTAMP = True
FROZEN = True
ORIGIN_DATASET = PROJECT_ROOT / "data" / "TaxolDataset"

NUM_EPOCHS = 40
DEVICE = torch.device("cuda")
N_WORKERS = 4
BATCH_SIZE = 256
WITH_AUGS = False
DEBUG = False
FIXED_SPLIT = False
BACKBONE = "resnet152"
