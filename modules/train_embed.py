import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import wandb
from catalyst.dl.runner import SupervisedWandbRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.utils import split_dataframe_train_test

from modules.data import get_data, get_loaders_siamese
from modules.losses import ContrastiveLoss
from modules.models import ResNetEncoder, SiameseNet


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
    print("Data dir", config.DATA_DIR)
    df_with_labels, class_names, num_classes = get_data(config.DATA_DIR, config.MODE)
    print("Dataset size", df_with_labels.shape[0])

    train_data, valid_data = split_dataframe_train_test(df_with_labels, test_size=0.2,  # TODO 100 of each class for test
                                                        random_state=config.SEED)
    wandb.config.update({"train_size": train_data.shape[0], "valid_size": valid_data.shape[0]})

    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict(
        'records')

    loaders = get_loaders_siamese(data_dir=config.DATA_DIR,
                                  train_data=train_data,
                                  valid_data=valid_data,
                                  batch_size=config.BATCH_SIZE,
                                  num_workers=4,
                                  transforms=config.TRANSFORMS)

    encoder = ResNetEncoder(pretrained=False)
    model = SiameseNet(encoder)

    criterion = ContrastiveLoss(margin=1.)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = None
    runner = SupervisedWandbRunner(device=config.DEVICE)

    runner.train(
        model=model,
        logdir=log_dir,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=config.NUM_EPOCHS,
        verbose=True,
        fp16={"apex": False}
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=Path, required=True)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_known_args()[0]

    shutil.copy2(args.config_path, "/project/modules/config.py")  # TODO

    main()
