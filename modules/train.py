import os
import shutil

import torch
import ttach as tta
import wandb
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback
from catalyst.utils import set_global_seed, prepare_cudnn
from sklearn.metrics import f1_score, accuracy_score
from torch import nn

from modules.callbacks import ConfusionMatrixCallback, MissCallback, WandbCallback, \
    BestMetricAccumulator
from modules.data import get_loaders, get_frozen_transforms, get_transforms, get_test_loader, train_val_test_split
from modules.models import ClassificationNet, LargerClassificationNet
from modules.utils import get_nn_parser


def main(config):
    set_global_seed(config.seed)
    prepare_cudnn(deterministic=True)

    if config.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(project="microtubules_classification")
    wandb.run.save()

    wandb.config.update(args)
    wandb.config.update(dict(model="NN", mode=config.mode.value),
                        allow_val_change=True)

    log_dir = config.log_root / wandb.run.name
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.config, str(log_dir))

    train_df, valid_df, test_df, num_classes, class_names = \
        train_val_test_split(data_dir=config.data_root / config.data,
                             mode=config.mode, seed=config.seed,
                             split_path=config.split_path)

    if config.debug:
        train_df, valid_df = train_df[:config.batch_size], valid_df[:config.batch_size]
        test_df = test_df[:config.batch_size]

    wandb.config.update({"train_size": train_df.shape[0], "valid_size": valid_df.shape[0]})

    train_data, valid_data = train_df.to_dict('records'), valid_df.to_dict('records')
    test_data = test_df.to_dict('records')

    transforms = get_transforms() if config.with_augs else get_frozen_transforms()
    train_val_loaders = get_loaders(data_dir=config.data_root / config.data,
                                    train_data=train_data,
                                    valid_data=valid_data,
                                    num_classes=num_classes,
                                    num_workers=config.n_workers,
                                    batch_size=config.batch_size,
                                    transforms=transforms,
                                    shuffle_train=True)
    test_loader = get_test_loader(data_dir=config.data_root / config.data, test_data=test_data,
                                  num_classes=num_classes, num_workers=config.n_workers)

    model_fn = {1: ClassificationNet,
                2: LargerClassificationNet}[config.n_layers]

    model = model_fn(backbone_name=config.backbone,
                     n_classes=num_classes)
    model = model.to(config.device)
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
        loaders=train_val_loaders,
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

    ckpth_path = log_dir / "checkpoints" / "best.pth"
    runner = SupervisedRunner(device=config.device)
    test_logits = runner.predict_loader(model, loader=test_loader,
                                        resume=ckpth_path,
                                        verbose=True)
    y_test_pred = test_logits.argmax(1)
    y_test = valid_df.label.to_numpy()

    fscore = f1_score(y_test, y_test_pred, average='macro')
    acc = accuracy_score(y_test, y_test_pred)

    wandb.log({"test/fscore": fscore, "test/acc": acc})
    wandb.log(commit=True)

    print("fscore", fscore)
    print("acc", acc)


if __name__ == '__main__':
    parser = get_nn_parser()
    args, _ = parser.parse_known_args()
    if args.fixed_split and args.split_path is None:
        parser.error("--fixed_split requires --split_path")

    main(args)
