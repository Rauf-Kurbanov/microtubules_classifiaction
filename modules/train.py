from pathlib import Path

import pretrainedmodels
import torch
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback
from catalyst.dl.runner import SupervisedRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.utils.dataset import split_dataframe
from torch import nn
from datetime import datetime

from modules.data import get_loaders, get_data


def get_model(model_name: str, num_classes: int, pretrained: str = "imagenet"):
    model_fn = pretrainedmodels.__dict__[model_name]
    model = model_fn(num_classes=1000, pretrained=pretrained)

    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)

    return model


def main():
    SEED = 42
    set_global_seed(SEED)
    prepare_cudnn(deterministic=True)

    DATA_DIR = Path("/data/NewArchive")
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    LOG_ROOT = Path("/data/logs")
    log_dir = LOG_ROOT / f"tubles_{current_time}"
    NUM_EPOCHS = 15

    df_with_labels, class_names, num_classes = get_data(DATA_DIR)
    train_data, valid_data = split_dataframe(df_with_labels, test_size=0.2,
                                             random_state=SEED)
    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict(
        'records')

    loaders = get_loaders(data_dir=DATA_DIR,
                          train_data=train_data,
                          valid_data=valid_data,
                          num_classes=num_classes)

    model_name = "resnet18"
    model = get_model(model_name, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[9], gamma=0.3
    )

    runner = SupervisedRunner()

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
            )
        ],
        num_epochs=NUM_EPOCHS,
        verbose=True
    )


if __name__ == '__main__':
    main()
