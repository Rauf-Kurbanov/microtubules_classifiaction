import argparse
import importlib
import shutil
from datetime import datetime
from pathlib import Path

import torch
import wandb
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.utils import split_dataframe_train_test
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from torch import nn

from modules.data import get_loaders, get_data, get_frozen_transforms, get_transforms
from modules.models import ResNetEncoder


class FeatureExtractor(nn.Module):  # TODO
    def __init__(self, embed_net, n_classes):
        super().__init__()
        self.embedding_net = embed_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        return output


def main(config):
    set_global_seed(config.SEED)
    prepare_cudnn(deterministic=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    frozen_tag = "FROZEN" if config.FROZEN else ""
    timestamp = str(current_time) if config.WITH_TIMESTAMP else ""
    run_name = f"{config.DATA_DIR.stem}_{config.MODE.name}_{frozen_tag}_{timestamp}"

    log_dir = Path(config.LOG_ROOT / run_name)
    wandb.init(project="microtubules_classification")
    wandb.config.data = config.DATA_DIR.name
    wandb.config.mode = config.MODE.name
    wandb.config.model = "SVM"
    wandb.config.seed = config.SEED
    wandb.config.with_augs = config.WITH_AUGS
    wandb.config.debug = config.DEBUG

    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.__file__, str(log_dir))
    df_with_labels, class_names, num_classes = get_data(config.DATA_DIR, config.MODE)

    train_data, valid_data = split_dataframe_train_test(df_with_labels, test_size=0.2,
                                                        random_state=config.SEED)
    if config.DEBUG:
        train_data, valid_data = train_data[:config.BATCH_SIZE], valid_data[:config.BATCH_SIZE]

    wandb.config.update({"train_size": train_data.shape[0], "valid_size": valid_data.shape[0]})

    train_data, valid_data = train_data.to_dict('records'), valid_data.to_dict(
        'records')

    print("Train size:", len(train_data))
    print("Valid size:", len(valid_data))

    transforms = get_transforms() if config.WITH_AUGS else get_frozen_transforms()
    loaders = get_loaders(data_dir=config.DATA_DIR,
                          train_data=train_data,
                          valid_data=valid_data,
                          num_classes=num_classes,
                          num_workers=config.N_WORKERS,
                          batch_size=config.BATCH_SIZE,
                          transforms=transforms)

    encoder = ResNetEncoder(frozen=config.FROZEN)

    model = FeatureExtractor(embed_net=encoder,
                             n_classes=num_classes)
    model = model.to(config.DEVICE)

    X, y = dict(), dict()
    for loader_name, loader in loaders.items():
        xs, ys = [], []
        for data_dict in loader:
            data = model(data_dict['features'].to(config.DEVICE))
            label = data_dict['targets']
            xs.append(data)
            ys.append(label)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        X[loader_name] = xs.detach().cpu().numpy()
        y[loader_name] = ys.detach().cpu().numpy()

    clf = SVC()

    X_train, y_train = X['train'], y['train']
    X_valid, y_valid = X['valid'], y['valid']

    clf.fit(X_train, y_train)

    y_valid_pred = clf.predict(X_valid)
    y_train_pred = clf.predict(X_train)

    fscore = f1_score(y_valid, y_valid_pred, average='macro')
    acc = accuracy_score(y_valid, y_valid_pred)

    train_fscore = f1_score(y_train, y_train_pred, average='macro')
    train_acc = accuracy_score(y_train, y_train_pred)

    print("TRAIN")
    print("Accuracy:", train_acc)
    print("F1 score:", train_fscore)

    print("VALID")
    print("Accuracy:", acc)
    print("F1 score:", fscore)

    wandb.log({"best/accuracy01/best_epoch": acc,
               "best/f1_score/best_epoch": fscore})


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_known_args()[0]

    config = importlib.import_module(f"configs.embed.{args.config_name}")
    main(config)
