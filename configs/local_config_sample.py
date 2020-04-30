from pathlib import Path

import torch

from modules.utils import Mode

SEED = 42
PROJECT_ROOT = Path("/Users/raufkurbanov/Programs/microtubules_classifiaction")
DATA_DIR = PROJECT_ROOT / "data" / "Processed"
# DATA_DIR = PROJECT_ROOT / "data" / "Processed"
LOG_ROOT = PROJECT_ROOT / "results" / "logs"
MODE = Mode.ZERO_VS_ZERO_ONE_VS_ONE
WITH_TIMESTAMP = True
FROZEN = True
MAIN_METRIC = "accuracy01"  # "auc/_mean"

NUM_EPOCHS = 10
DEVICE = torch.device("cpu")
N_WORKERS = 1
BATCH_SIZE = 64
WITH_AUGS = True
DEBUG = True
SIAMESE_CKPT = None
ORIGIN_DATASET = PROJECT_ROOT / "data" / "TaxolDataset"
