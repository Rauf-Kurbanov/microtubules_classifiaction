import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pretrainedmodels
import pretrainedmodels.utils as putils
import wandb
from catalyst.dl import SupervisedRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from skimage import io
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC

from modules.data import get_loaders, get_frozen_transforms, get_transforms, get_test_loader, train_val_test_split
from modules.utils import fig_to_pil, get_svm_parser


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

    if config.wandb_offline:  # TODO creates empty runs
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(project="microtubules_classification")
    wandb.run.save()

    wandb.config.update(args)
    wandb.config.update(dict(model="SVM", mode=config.mode.value),
                        allow_val_change=True)

    log_dir = config.log_root / wandb.run.name
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.config, str(log_dir))

    train_df, valid_df, test_df, num_classes, _ = train_val_test_split(data_dir=config.data_dir,
                                                                    mode=config.mode, seed=config.seed,
                                                                    split_path=config.split_path)

    if config.debug:
        train_df, valid_df, test_df = \
            train_df[:config.batch_size], valid_df[:config.batch_size], test_df[:config.batch_size]

    wandb.config.update({"train_size": train_df.shape[0], "valid_size": test_df.shape[0]})

    train_data, valid_data, test_data = \
        train_df.to_dict('records'), valid_df.to_dict('records'), test_df.to_dict('records')

    test_loader = get_test_loader(data_dir=config.data_dir, test_data=test_data,
                                  num_classes=num_classes, num_workers=config.n_workers)

    transforms = get_transforms() if config.with_augs else get_frozen_transforms()

    model = pretrainedmodels.__dict__[config.backbone]()
    model.last_linear = putils.Identity()
    model = model.to(config.device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    runner = SupervisedRunner(device=config.device)
    model.last_linear = putils.Identity()

    train_val_loaders = get_loaders(data_dir=config.data_dir,
                                    train_data=train_data,
                                    valid_data=valid_data,
                                    num_classes=num_classes,
                                    num_workers=config.n_workers,
                                    batch_size=config.batch_size,
                                    shuffle_train=False,
                                    transforms=transforms)
    X_train = runner.predict_loader(model, loader=train_val_loaders["train"],
                                    verbose=True)
    X_test = runner.predict_loader(model, loader=test_loader,
                                   verbose=True)

    clf = SVC()
    y_train = train_df.label.to_numpy()
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)
    y_test = test_df.label.to_numpy()

    fscore = f1_score(y_test, y_test_pred, average='macro')
    acc = accuracy_score(y_test, y_test_pred)
    print("fscore", fscore)
    print("acc", acc)

    # train_fscore = f1_score(y_train, y_train_pred, average='macro')
    # train_acc = accuracy_score(y_train, y_train_pred)

    wandb.log({"best/accuracy01/best_epoch": acc,
               "best/f1_score/best_epoch": fscore})

    # wandb.sklearn.plot_confusion_matrix(y_valid, y_valid_pred, class_names)

    # missclassified_ = metadata['valid']  # TODO pred target
    # missclassified_ = [{'name': d["name"], 'pred': pred, 'target': d['label']}
    #                    for d, pred in zip(missclassified_, list(y_valid_pred))]  # TODO filter
    # missclassified_ = [d for d in missclassified_ if d['pred'] != d['target']]  # TODO filter
    # on_epoch_end(missclassified_, _class_names=class_names,
    #              origin_ds=config.data_root / config.origin_data, n_examples=5)
    #


if __name__ == '__main__':
    parser = get_svm_parser()
    args, _ = parser.parse_known_args()
    if args.fixed_split and args.split_path is None:
        parser.error("--fixed_split requires --split_path")

    main(args)

