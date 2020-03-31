import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pretrainedmodels
import torch
import wandb
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback
from catalyst.dl.runner import SupervisedWandbRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.utils import split_dataframe_train_test
from torch import nn

from modules.callbacks import ConfusionMatrixCallback, EmbedPlotCallback
from modules.data import get_loaders, get_data


def get_model(model_name: str, num_classes: int, pretrained: str = "imagenet",
              frozen_encoder=True):
    model_fn = pretrainedmodels.__dict__[model_name]
    model = model_fn(num_classes=1000, pretrained=pretrained)
    if frozen_encoder:
        for param in model.parameters():
            param.requires_grad = False

    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)

    return model


def main():
    from modules import config

    set_global_seed(config.SEED)
    prepare_cudnn(deterministic=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    frozen_tag = "FROZEN" if config.FROZEN else ""
    timestamp = str(current_time) if config.WITH_TIMESTAMP else ""
    run_name = f"{config.DATA_DIR.stem}_{config.MODE.name}_{frozen_tag}_{timestamp}"

    log_dir = Path(config.LOG_ROOT / run_name)
    wandb.init(project="microtubules_classification", name=run_name)
    wandb.config.batch_size = config.BATCH_SIZE
    wandb.config.data = config.DATA_DIR.name
    wandb.config.mode = config.MODE.name
    wandb.config.frozen = True
    wandb.config.seed = config.SEED

    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.__file__, str(log_dir))
    df_with_labels, class_names, num_classes = get_data(config.DATA_DIR, config.MODE)

    train_data, valid_data = split_dataframe_train_test(df_with_labels, test_size=0.2,  # TODO 100 of each class for test
                                                        random_state=config.SEED)
    wandb.config.update({"train_size": train_data.shape[0], "valid_size": valid_data.shape[0]})

    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict(
        'records')

    loaders = get_loaders(data_dir=config.DATA_DIR,
                          train_data=train_data,
                          valid_data=valid_data,
                          num_classes=num_classes,
                          num_workers=4,
                          batch_size=config.BATCH_SIZE)

    model_name = "resnet18"
    model = get_model(model_name, num_classes, frozen_encoder=config.FROZEN)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[9], gamma=0.3  # TODO milestones ??
    # )
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = None
    # runner = SupervisedRunner(device=config.DEVICE)
    runner = SupervisedWandbRunner(device=config.DEVICE)

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
            EmbedPlotCallback(config.MODE)
        ],
        num_epochs=config.NUM_EPOCHS,
        verbose=True,
        main_metric=config.MAIN_METRIC,
        minimize_metric=False,
        fp16={"apex": False}
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, required=True)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_known_args()[0]
    shutil.copy2(args.config_path, "/project/modules/config.py")

    main()
