import io
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pretrainedmodels
import torch
from PIL import Image
from torch import nn
from torch.nn.functional import softmax
import configargparse


class Mode(Enum):  # TODO move
    ZERO_VS_ZERO_ONE = "0_vs_0.1"
    ZERO_VS_ONE = "0_vs_1"
    ZERO_ONE_VS_ONE = "0.1_vs_1"
    ZERO_VS_ZERO_ONE_VS_ONE = "0_vs_0.1_vs_1"


def get_model(model_name: str, num_classes: int, pretrained: str = "imagenet"):
    model_fn = pretrainedmodels.__dict__[model_name]
    model = model_fn(num_classes=1000, pretrained=pretrained)

    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)

    return model


def show_examples(images: List[Tuple[str, np.ndarray]]):
    _indexes = [(i, j) for i in range(2) for j in range(2)]

    f, ax = plt.subplots(2, 2, figsize=(16, 16))
    for (i, j), (title, img) in zip(_indexes, images):
        ax[i, j].imshow(img)
        ax[i, j].set_title(title)
    f.tight_layout()


def read_random_images(paths: List[Path]) -> List[Tuple[str, np.ndarray]]:
    data = np.random.choice(paths, size=4)
    result = []
    for d in data:
        title = f"{d.parent.name}: {d.name}"
        _image = plt.imread(d)
        result.append((title, _image))

    return result


def show_prediction(
        model: torch.nn.Module,
        valid_transforms,
        class_names: List[str],
        titles: List[str],
        images: List[np.ndarray],
        device: torch.device
) -> None:
    with torch.no_grad():
        tensor_ = torch.stack([
            valid_transforms(image=image)["image"]
            for image in images
        ]).to(device)

        logits = model.forward(tensor_)
        probabilities = softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    images_predicted_classes = [
        (f"predicted: {class_names[x]} | correct: {title}", image)
        for x, title, image in zip(predictions, titles, images)
    ]
    show_examples(images_predicted_classes)


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    pil_img = deepcopy(Image.open(buf))
    buf.close()

    return pil_img


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def get_nn_parser():
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", required=True, is_config_file=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--data", type=str) # TODO
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
    parser.add_argument("--wandb_offline", type=str2bool)

    return parser


def get_svm_parser():
    parser = configargparse.ArgParser()
    parser.add_argument("-c", "--config", required=True, is_config_file=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--log_root", type=Path)
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
    parser.add_argument("--wandb_offline", type=str2bool)

    return parser