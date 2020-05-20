import collections

import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, VerticalFlip
from albumentations import LongestMaxSize, PadIfNeeded
from albumentations import Normalize
from albumentations import ShiftScaleRotate, IAAPerspective, RandomBrightnessContrast, \
    RandomGamma, HueSaturationValue, ImageCompression
from albumentations.pytorch import ToTensorV2
from catalyst.data import ImageReader
from catalyst.data.augmentor import Augmentor
from catalyst.data.reader import ScalarReader, ReaderCompose
from catalyst.dl import utils
from catalyst.dl.utils import get_loader
from catalyst.utils import create_dataset, create_dataframe, map_dataframe
from catalyst.utils import get_dataset_labeling
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate as default_collate_fn
from torchvision import transforms

from modules.utils import Mode

BORDER_CONSTANT = 0
BORDER_REFLECT = 2


class TifImageReader(ImageReader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, row):
        image_name = str(row[self.input_key])
        img = plt.imread(image_name)

        result = {self.output_key: img, 'original': img, 'name': image_name}
        return result


def pre_transforms(image_size=224):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        LongestMaxSize(max_size=image_size),
        PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT)
    ]

    return result


def hard_transforms():
    result = [
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=BORDER_REFLECT,
            p=0.5
        ),
        IAAPerspective(scale=(0.02, 0.05), p=0.3),
        RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        RandomGamma(gamma_limit=(85, 115), p=0.3),
        HueSaturationValue(p=0.3),
        ImageCompression(quality_lower=80),
        HorizontalFlip(p=0.3),
        VerticalFlip(p=0.3)
    ]

    return result


def post_transforms():
    return [Normalize(), ToTensorV2()]


def compose(_transforms):
    result = Compose([item for sublist in _transforms for item in sublist])
    return result


def get_transforms():
    no_transforms = compose([pre_transforms()])
    train_transforms = compose(
        [pre_transforms(), hard_transforms(), post_transforms()])
    valid_transforms = compose([pre_transforms(), post_transforms()])

    train_data_transforms = transforms.Compose([
        Augmentor(
            dict_key="features",
            augment_fn=lambda x: train_transforms(image=x)["image"]
        ),
        Augmentor(
            dict_key="original",
            augment_fn=lambda x: no_transforms(image=x)["image"]
        )
    ])

    valid_data_transforms = transforms.Compose([
        Augmentor(
            dict_key="features",
            augment_fn=lambda x: valid_transforms(image=x)["image"]
        ),
        Augmentor(
            dict_key="original",
            augment_fn=lambda x: no_transforms(image=x)["image"]
        )
    ])

    return train_data_transforms, valid_data_transforms


def get_frozen_transforms():
    no_transforms = compose([pre_transforms()])
    valid_transforms = compose([pre_transforms(), post_transforms()])

    train_data_transforms = transforms.Compose([
        Augmentor(
            dict_key="features",
            augment_fn=lambda x: valid_transforms(image=x)["image"]
        ),
        Augmentor(
            dict_key="original",
            augment_fn=lambda x: no_transforms(image=x)["image"]
        )

    ])

    valid_data_transforms = transforms.Compose([
        Augmentor(
            dict_key="features",
            augment_fn=lambda x: valid_transforms(image=x)["image"]
        ),
        Augmentor(
            dict_key="original",
            augment_fn=lambda x: no_transforms(image=x)["image"]
        )

    ])

    return train_data_transforms, valid_data_transforms


def get_loaders(*,
                data_dir,
                train_data, valid_data,
                num_classes,
                batch_size: int = 64,
                num_workers: int = 4,
                sampler=None,
                transforms=None) -> collections.OrderedDict:
    open_fn = ReaderCompose([
        TifImageReader(
            input_key="filepath",
            output_key="features",
            rootpath=data_dir
        ),

        ScalarReader(
            input_key="label",
            output_key="targets",
            default_value=-1,
            dtype=np.int64
        ),

        ScalarReader(
            input_key="label",
            output_key="targets_one_hot",
            default_value=-1,
            dtype=np.int64,
            one_hot_classes=num_classes
        )
    ])

    if transforms is None:
        train_data_transforms, valid_data_transforms = get_transforms()
    else:
        train_data_transforms, valid_data_transforms = transforms

    train_loader = get_loader(
        train_data,
        open_fn=open_fn,
        dict_transform=train_data_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=sampler is None,
        # shuffle data only if Sampler is not specified (PyTorch requirement)
        sampler=sampler,
        drop_last=True
    )

    valid_loader = utils.get_loader(
        valid_data,
        open_fn=open_fn,
        dict_transform=valid_data_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=None
    )

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


def _get_data(data_dir):
    dataset = create_dataset(dirs=f"{data_dir}/*", extension="*.tif")
    df = create_dataframe(dataset, columns=["class", "filepath"])
    tag_to_label = get_dataset_labeling(df, "class")
    df_with_labels = map_dataframe(df, tag_column="class", class_column="label",
                                   tag2class=tag_to_label, verbose=True)

    class_names = [name for name, id_ in
                   sorted(tag_to_label.items(), key=lambda x: x[1])]
    num_classes = len(tag_to_label)

    return df_with_labels, class_names, num_classes


def filter_data_by_mode(df_with_labels, class_names, num_classes, mode):
    if mode is Mode.ZERO_VS_ZERO_ONE:
        df_with_labels.loc[df_with_labels["class"] == "Control", "label"] = 0
        df_with_labels.loc[df_with_labels["class"] == "01Taxol", "label"] = 1
        df_with_labels = df_with_labels[df_with_labels['class'] != "1Taxol"]
        return df_with_labels, ["Control", "01Taxol"], 2

    if mode is Mode.ZERO_VS_ONE:
        df_with_labels.loc[df_with_labels["class"] == "Control", "label"] = 0
        df_with_labels.loc[df_with_labels["class"] == "1Taxol", "label"] = 1
        df_with_labels = df_with_labels[df_with_labels['class'] != "01Taxol"]
        return df_with_labels, ["Control", "1Taxol"], 2

    if mode is Mode.ZERO_ONE_VS_ONE:
        df_with_labels.loc[df_with_labels["class"] == "01Taxol", "label"] = 0
        df_with_labels.loc[df_with_labels["class"] == "1Taxol", "label"] = 1
        df_with_labels = df_with_labels[df_with_labels['class'] != "Control"]
        return df_with_labels, ["01Taxol", "1Taxol"], 2

    return df_with_labels, class_names, num_classes


class SiameseDataset(Dataset):

    def __init__(self, *, data_dir, train_data, is_train, transform, random_seed=42):
        self.train = is_train
        self.transform = transform

        self.open_fn = TifImageReader(
            input_key="filepath",
            output_key="features",
            rootpath=data_dir)

        self.labels = np.array([d["label"] for d in train_data])
        self.filepaths = [d["filepath"] for d in train_data]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}

        if not is_train:
            random_state = np.random.RandomState(random_seed)
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.labels[i]]),
                               1]
                              for i in range(0, len(self.filepaths), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - {self.labels[i]})
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.filepaths), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.filepaths[index], self.labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - {label1}))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.filepaths[siamese_index]
        else:
            img1 = self.filepaths[self.test_pairs[index][0]]
            img2 = self.filepaths[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = self.open_fn({"filepath": img1})
        img2 = self.open_fn({"filepath": img2})

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        img1 = img1["features"]  # TODO
        img2 = img2["features"]  # TODO
        return {"features": (img1, img2), "targets": target}

    def __len__(self):
        return len(self.filepaths)


def get_loaders_siamese(*,
                        data_dir,
                        train_data, valid_data,
                        batch_size: int = 64,
                        num_workers: int = 4,
                        transforms=None) -> collections.OrderedDict:

    if transforms is None:  # TODO
        train_data_transforms, valid_data_transforms = get_transforms()
    else:
        train_data_transforms, valid_data_transforms = transforms

    train_dataset = SiameseDataset(data_dir=data_dir,
                                   train_data=train_data,
                                   is_train=True,
                                   transform=train_data_transforms)
    valid_dataset = SiameseDataset(data_dir=data_dir,
                                   train_data=valid_data,
                                   is_train=True,
                                   transform=train_data_transforms)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        collate_fn=default_collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        collate_fn=default_collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders
