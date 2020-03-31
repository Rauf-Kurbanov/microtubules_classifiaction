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


class Mode(Enum):
    ZERO_VS_ZERO_ONE = 0
    ZERO_VS_ONE = 1
    ZERO_ONE_VS_ONE = 2
    ZERO_VS_ZERO_ONE_VS_ONE = 3


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
