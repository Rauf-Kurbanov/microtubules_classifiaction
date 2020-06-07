import argparse
import shutil
from pathlib import Path

import configargparse
import pandas as pd
import torch
import ttach as tta
import wandb
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.utils import split_dataframe_train_test
from torch import nn

from modules.callbacks import ConfusionMatrixCallback, MissCallback, WandbCallback, \
    BestMetricAccumulator
from modules.data import get_loaders, _get_data, get_frozen_transforms, get_transforms, filter_data_by_mode
from modules.models import ClassificationNet, LargerClassificationNet
from modules.utils import Mode, str2bool


def main(config):
    set_global_seed(config.seed)
    prepare_cudnn(deterministic=True)

    wandb.init(project="microtubules_classification")
    wandb.run.save()

    log_dir = Path(config.log_root / wandb.run.name)

    # TODO
    wandb.config.update(args)
    wandb.config.update(dict(model="NN", mode=config.mode.value),
                        allow_val_change=True)

    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.config, str(log_dir))
    df_with_labels, class_names, num_classes = _get_data(config.data_root / config.data)

    if config.fixed_split:
        # TODO refactor
        train_data = pd.read_csv(config.split_path / "train.csv",
                                 usecols=["class", "filepath", "label"])
        valid_data = pd.read_csv(config.split_path / "test.csv",
                                 usecols=["class", "filepath", "label"])
        train_data.filepath = train_data.filepath.apply(lambda p: config.data_root / p)
        valid_data.filepath = valid_data.filepath.apply(lambda p: config.data_root / p)
        train_data, class_names, num_classes = filter_data_by_mode(train_data, class_names, num_classes, config.mode)
        valid_data, class_names, num_classes = filter_data_by_mode(valid_data, class_names, num_classes, config.mode)
    else:
        df_with_labels, class_names, num_classes = filter_data_by_mode(df_with_labels, class_names, num_classes,
                                                                       config.mode)
        train_data, valid_data = split_dataframe_train_test(df_with_labels, test_size=0.2,
                                                            random_state=config.seed)

    if config.debug:
        train_data, valid_data = train_data[:config.batch_size], valid_data[:config.batch_size]

    wandb.config.update({"train_size": train_data.shape[0], "valid_size": valid_data.shape[0]})

    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict(
        'records')

    transforms = get_transforms() if config.with_augs else get_frozen_transforms()
    loaders = get_loaders(data_dir=config.data_root / config.data,
                          train_data=train_data,
                          valid_data=valid_data,
                          num_classes=num_classes,
                          num_workers=config.n_workers,
                          batch_size=config.batch_size,
                          transforms=transforms)

    model_fn = {1: ClassificationNet,
                2: LargerClassificationNet}[config.n_layers]

    model = model_fn(backbone_name=config.backbone,
                     n_classes=num_classes)
    wandb.watch(model, log_freq=20)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = None
    runner = SupervisedRunner(device=config.device)

    if config.tta:
        model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform())

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
            ConfusionMatrixCallback(config.mode),
            # EmbedPlotCallback(config.mode),
            MissCallback(config.mode, origin_ds=config.data_root / config.origin_data),
            WandbCallback(),
            BestMetricAccumulator(),
        ],
        num_epochs=config.num_epoch,
        verbose=True,
        main_metric=config.main_metric,
        minimize_metric=False,
        fp16={"apex": False}
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    return parser


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", required=True, is_config_file=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--data", type=str)
    parser.add_argument("--log_root", type=Path)
    parser.add_argument("--project_root", type=Path)
    parser.add_argument("--mode", type=Mode, choices=list(Mode))
    parser.add_argument("--frozen", type=str2bool)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--device", type=torch.device)
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--with_augs", type=str2bool)
    parser.add_argument("--debug", type=str2bool)
    parser.add_argument("--origin_data", type=str)
    parser.add_argument("--fixed_split", type=str2bool)
    parser.add_argument("--split_path", type=Path)
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--tta", type=str2bool)
    parser.add_argument("--n_layers", type=int, choices=[1, 2])
    parser.add_argument("--main_metric", type=str,
                        help="the key to the name of the metric by which the checkpoints will be selected")

    args, _ = parser.parse_known_args()
    if args.fixed_split and args.split_path is None:
        parser.error("--fixed_split requires --split_path")

    main(args)
