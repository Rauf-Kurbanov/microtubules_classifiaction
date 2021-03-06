import argparse
import random
from pathlib import Path

import configargparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretrainedmodels
import pretrainedmodels.utils as putils
import torch
import wandb
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.utils import split_dataframe_train_test
from skimage import io
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC

from modules.data import get_loaders, _get_data, get_frozen_transforms, get_transforms, filter_data_by_mode
from modules.utils import fig_to_pil, str2bool, Mode


def on_epoch_end(_missclassified, _class_names, origin_ds, n_examples=5):

    n_to_sample = min(n_examples, len(_missclassified))
    miss_sample = random.sample(_missclassified, n_to_sample)

    fig, axes = plt.subplots(nrows=3, ncols=n_to_sample, figsize=(30, 15))
    if n_to_sample == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(n_to_sample):
        miss_d = miss_sample[i]
        image = io.imread(Path(miss_d['name']), as_gray=True)
        axes[0, i].imshow(image, cmap=plt.cm.gray)
        axes[0, 0].set_ylabel('Cropped')
        axes[0, i].set_title(f'{_class_names[miss_d["pred"]]} instead of {_class_names[miss_d["target"]]}')
        p = Path(miss_d['name'])
        axes[0, i].set_xlabel(f'{p.relative_to(p.parent.parent.parent)}')

        t_image = image
        m = cv2.UMat(np.zeros_like(t_image))
        ntimage = cv2.normalize(t_image, m, 0, 255, cv2.NORM_MINMAX).get()
        ntimage = ntimage.astype(np.int)
        axes[1, i].imshow(ntimage, cmap=plt.cm.gray)
        axes[1, 0].set_ylabel('Transformed')
        axes[2, 0].set_ylabel('Origin')

        def _drop_last_suffix(name):
            p = Path(name)
            name_parts = p.name.split('_')
            ext = name_parts[-1].split('.')[1]
            new_name = f"{'_'.join(name_parts[:-1])}.{ext}"
            return new_name

        new_name = _drop_last_suffix(miss_d['name'])
        p = Path(miss_d['name'])
        class_name = p.parent.stem
        origin_file = origin_ds / class_name / new_name
        origin_img = plt.imread(origin_file)
        axes[2, i].imshow(origin_img, cmap=plt.cm.gray)
        axes[2, i].set_xlabel(f'{origin_file.relative_to(origin_file.parent.parent.parent)}')

    wandb.log({"Miss Examples":  [wandb.Image(fig_to_pil(fig), caption="Label")]})
    _missclassified = []


def main(config):
    set_global_seed(config.seed)
    prepare_cudnn(deterministic=True)

    wandb.init(project="microtubules_classification")
    wandb.run.save()

    wandb.config.update(args)
    wandb.config.update(dict(model="SVM", mode=config.mode.value),
                        allow_val_change=True)

    df_with_labels, class_names, num_classes = _get_data(config.data_root / config.data)
    if config.fixed_split:
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

    model = pretrainedmodels.__dict__[config.backbone]()
    model.last_linear = putils.Identity()
    model = model.to(config.device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    X, y, metadata = dict(), dict(), dict()
    for loader_name, loader in loaders.items():
        xs, ys = [], []
        meta = []
        for data_dict in loader:
            data = model(data_dict['features'].to(config.device))
            data = data.to(torch.device("cpu"))
            label = data_dict['targets']
            xs.append(data)
            ys.append(label)
            for l, n, o in zip(label, data_dict["name"], data_dict["original"]):
                meta.append({"label": l.item(), "name": n})
            if config.debug:
                break
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        X[loader_name] = xs.detach().cpu().numpy()
        y[loader_name] = ys.detach().cpu().numpy()
        metadata[loader_name] = meta

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

    wandb.log({"best/accuracy01/best_epoch": acc,
               "best/f1_score/best_epoch": fscore})

    wandb.sklearn.plot_confusion_matrix(y_valid, y_valid_pred, class_names)

    missclassified_ = metadata['valid']  # TODO pred target
    missclassified_ = [{'name': d["name"], 'pred': pred, 'target': d['label']}
                       for d, pred in zip(missclassified_, list(y_valid_pred))]  # TODO filter
    missclassified_ = [d for d in missclassified_ if d['pred'] != d['target']]  # TODO filter
    on_epoch_end(missclassified_, _class_names=class_names,
                 origin_ds=config.data_root / config.origin_data, n_examples=5)


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
    parser.add_argument("--project_root", type=Path)
    parser.add_argument("--mode", type=Mode, choices=list(Mode))
    parser.add_argument("--device", type=torch.device)
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--with_augs", type=str2bool)
    parser.add_argument("--debug", type=str2bool)
    parser.add_argument("--origin_data", type=str)
    parser.add_argument("--fixed_split", type=str2bool)
    parser.add_argument("--split_path", type=Path)
    parser.add_argument("--backbone", type=str)

    args, _ = parser.parse_known_args()
    if args.fixed_split and args.split_path is None:
        parser.error("--fixed_split requires --split_path")

    main(args)
