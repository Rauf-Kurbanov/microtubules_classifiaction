from pathlib import Path

import torch

from modules.utils import Mode

SEED = 42
PROJECT_ROOT = Path("/Users/raufkurbanov/Programs/microtubules_classifiaction")
DATASET_NAME = "FilteredCleanProcessed"
DATA_DIR = PROJECT_ROOT / "data" / DATASET_NAME
LOG_ROOT = PROJECT_ROOT / "results" / "logs"
MODE = Mode.ZERO_VS_ONE
WITH_TIMESTAMP = True
FROZEN = False

NUM_EPOCHS = 20
DEVICE = torch.device("cpu")
N_WORKERS = 4
BATCH_SIZE = 32
WITH_AUGS = True
DEBUG = True
ORIGIN_DATASET = PROJECT_ROOT / "data" / "TaxolDataset"
FIXED_SPLIT = False
