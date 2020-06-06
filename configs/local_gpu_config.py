from pathlib import Path

import torch

from modules.utils import Mode

SEED = 42
PROJECT_ROOT = Path("/project")
DATASET_NAME = "FilteredCleanProcessed"
DATA_DIR = PROJECT_ROOT / "data" / DATASET_NAME
LOG_ROOT = PROJECT_ROOT / "results" / "logs"
MODE = Mode.ZERO_VS_ONE
WITH_TIMESTAMP = True
FROZEN = True
MAIN_METRIC = "accuracy01"

NUM_EPOCHS = 40
DEVICE = torch.device("cuda")
N_WORKERS = 0
BATCH_SIZE = 256
WITH_AUGS = True

SIAMESE_CKPT = None

DEBUG = True
ORIGIN_DATASET = PROJECT_ROOT / "data" / "TaxolDataset"
FIXED_SPLIT = False
BACKBONE = "resnet18"
N_LAYERS = 2
TTA = True
