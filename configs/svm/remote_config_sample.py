from pathlib import Path

import torch

from modules.utils import Mode

SEED = 42
PROJECT_ROOT = Path("/project")
DATA_DIR = PROJECT_ROOT / "data" / "FilteredCleanProcessed"
LOG_ROOT = PROJECT_ROOT / "results" / "embed"
MODE = Mode.ZERO_ONE_VS_ONE
WITH_TIMESTAMP = True
FROZEN = True
# MAIN_METRIC = "accuracy01"  # "auc/_mean"
ORIGIN_DATASET = PROJECT_ROOT / "data" / "TaxolDataset"

NUM_EPOCHS = 20
DEVICE = torch.device("cuda")
N_WORKERS = 4
BATCH_SIZE = 512
WITH_AUGS = False
DEBUG = False
