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
FROZEN = True
MAIN_METRIC = "accuracy01"

NUM_EPOCHS = 10
DEVICE = torch.device("cpu")
N_WORKERS = 1
BATCH_SIZE = 64
WITH_AUGS = True
DEBUG = True
SIAMESE_CKPT = None
ORIGIN_DATASET = PROJECT_ROOT / "data" / "TaxolDataset"
FIXED_SPLIT = True
BACKBONE = "resnet18"
N_LAYERS = 2
