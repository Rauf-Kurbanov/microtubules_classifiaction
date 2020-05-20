import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import wandb
import importlib
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.utils import split_dataframe_train_test
from torch import nn
import pandas as pd

from modules.callbacks import ConfusionMatrixCallback, EmbedPlotCallback, MissCallback, WandbCallback, \
    BestMetricAccumulator
from modules.data import get_loaders, _get_data, get_frozen_transforms, get_transforms, filter_data_by_mode
from modules.models import ClassificationNet, ResNetEncoder, LargerClassificationNet


def main(config):
    set_global_seed(config.SEED)
    prepare_cudnn(deterministic=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    frozen_tag = "FROZEN" if config.FROZEN else ""
    timestamp = str(current_time) if config.WITH_TIMESTAMP else ""
    run_name = f"{config.DATA_DIR.stem}_{config.MODE.name}_{frozen_tag}_{timestamp}"

    log_dir = Path(config.LOG_ROOT / run_name)
    wandb.init(project="microtubules_classification")
    wandb.config.batch_size = config.BATCH_SIZE
    wandb.config.epochs = config.NUM_EPOCHS
    wandb.config.data = config.DATA_DIR.name
    wandb.config.mode = config.MODE.name
    wandb.config.frozen = config.FROZEN
    wandb.config.seed = config.SEED
    wandb.config.from_siamese = config.SIAMESE_CKPT is not None
    wandb.config.with_augs = config.WITH_AUGS
    wandb.config.debug = config.DEBUG
    wandb.config.fixed_split = config.FIXED_SPLIT

    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.__file__, str(log_dir))
    df_with_labels, class_names, num_classes = _get_data(config.DATA_DIR)

    if config.FIXED_SPLIT:
        train_data = pd.read_csv(config.PROJECT_ROOT / "data" / "splits" / config.DATASET_NAME / "train.csv",
                                 usecols=["class", "filepath", "label"])
        valid_data = pd.read_csv(config.PROJECT_ROOT / "data" / "splits" / config.DATASET_NAME / "test.csv",
                                usecols=["class", "filepath", "label"])
        train_data.filepath = train_data.filepath.apply(lambda p: config.PROJECT_ROOT / "data" / p)
        valid_data.filepath = valid_data.filepath.apply(lambda p: config.PROJECT_ROOT / "data" / p)
        train_data, class_names, num_classes = filter_data_by_mode(train_data, class_names, num_classes, config.MODE)
        valid_data, class_names, num_classes = filter_data_by_mode(valid_data, class_names, num_classes, config.MODE)
    else:
        df_with_labels, class_names, num_classes = filter_data_by_mode(df_with_labels, class_names, num_classes,
                                                                       config.MODE)
        train_data, valid_data = split_dataframe_train_test(df_with_labels, test_size=0.2,
                                                            random_state=config.SEED)

    if config.DEBUG:
        train_data, valid_data = train_data[:config.BATCH_SIZE], valid_data[:config.BATCH_SIZE]

    wandb.config.update({"train_size": train_data.shape[0], "valid_size": valid_data.shape[0]})

    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict(
        'records')

    transforms = get_transforms() if config.WITH_AUGS else get_frozen_transforms()
    loaders = get_loaders(data_dir=config.DATA_DIR,
                          train_data=train_data,
                          valid_data=valid_data,
                          num_classes=num_classes,
                          num_workers=config.N_WORKERS,
                          batch_size=config.BATCH_SIZE,
                          transforms=transforms)

    if config.SIAMESE_CKPT:
        encoder = ResNetEncoder.from_siamese_ckpt(config.SIAMESE_CKPT,
                                                  frozen=config.FROZEN)
    else:
        encoder = ResNetEncoder(frozen=config.FROZEN)

    wandb.config.model = "NN2"
    model = LargerClassificationNet(embed_net=encoder,
                              n_classes=num_classes)
    wandb.watch(model, log_freq=20)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = None
    runner = SupervisedRunner(device=config.DEVICE)

    runner.train(
        model=model,
        logdir=log_dir,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[
            AccuracyCallback(num_classes=num_classes),
            AUCCallback(
                num_classes=num_classes,
                input_key="targets_one_hot",
                class_names=class_names
            ),
            F1ScoreCallback(
                input_key="targets_one_hot",
                activation="Softmax"
            ),
            ConfusionMatrixCallback(config.MODE),
            # EmbedPlotCallback(config.MODE),
            MissCallback(config.MODE, origin_ds=config.ORIGIN_DATASET),
            WandbCallback(),
            BestMetricAccumulator()
        ],
        num_epochs=config.NUM_EPOCHS,
        verbose=True,
        main_metric=config.MAIN_METRIC,
        minimize_metric=False,
        fp16={"apex": False}
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_known_args()[0]

    config = importlib.import_module(f"configs.{args.config_name}")
    main(config)
