from itertools import chain
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm.auto import tqdm
import argparse


def bbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def process_folder(some_tif, to_folder,
                   volume_trs=30000, intensity_trs=50, peak_min_distance=55):

    image = plt.imread(some_tif)

    nimage = np.zeros_like(image)
    nimage = cv2.normalize(image,  nimage, 0, 255, cv2.NORM_MINMAX)
    fimage = nimage > intensity_trs

    distance = ndi.distance_transform_edt(fimage)
    local_maxi = peak_local_max(distance, min_distance=peak_min_distance, indices=False)

    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)

    for label in np.unique(labels):
        x0, x1, y0, y1 = bbox(labels == label)
        crop = nimage[x0:x1, y0:y1]
        volume = (labels == label).sum()

        if volume > volume_trs:
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(rgb_crop.astype(np.uint8))
            pil_img.save(to_folder / f'{some_tif.stem}_{label}.tif')


def process_dataset(from_folder, to_folder):
    tif_files = chain(from_folder.glob('**/*.tif'), from_folder.glob('**/*.tiff'))
    for p in tqdm(list(tif_files)):
        class_name = p.parent.name
        class_folder = to_folder / class_name
        class_folder.mkdir(exist_ok=True)
        process_folder(p, class_folder)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--from_folder', type=Path, default=Path('../data/TaxolDataset'))
    parser.add_argument('--to_folder', type=Path, default=Path('../data/Processed'))

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    process_dataset(args.from_folder, args.to_folder)
