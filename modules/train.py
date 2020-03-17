from datetime import datetime

import pretrainedmodels
import torch
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback
from catalyst.dl.runner import SupervisedRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.utils import split_dataframe_train_test
from torch import nn

from modules import config
from modules.data import get_loaders, get_data


def get_model(model_name: str, num_classes: int, pretrained: str = "imagenet"):
    model_fn = pretrainedmodels.__dict__[model_name]
    model = model_fn(num_classes=1000, pretrained=pretrained)

    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)

    return model


def main():
    set_global_seed(config.SEED)
    prepare_cudnn(deterministic=True)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = config.LOG_ROOT / f"tubles_{config.DATA_DIR.stem}_{config.MODE}_{current_time}"
    df_with_labels, class_names, num_classes = get_data(config.DATA_DIR, config.MODE)

    train_data, valid_data = split_dataframe_train_test(df_with_labels, test_size=0.2,  # TODO 100 of each class for test
                                                        random_state=config.SEED)
    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict(
        'records')

    loaders = get_loaders(data_dir=config.DATA_DIR,
                          train_data=train_data,
                          valid_data=valid_data,
                          num_classes=num_classes,
                          num_workers=0)

    model_name = "resnet18"
    model = get_model(model_name, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[9], gamma=0.3  # TODO milestones ??
    )

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
            )
        ],
        num_epochs=config.NUM_EPOCHS,
        verbose=True,
        main_metric="auc/_mean",
        fp16={"apex": False}
    )


if __name__ == '__main__':
    main()
