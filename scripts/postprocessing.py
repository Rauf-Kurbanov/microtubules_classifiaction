import argparse
import shutil
from pathlib import Path

import pandas as pd
from skimage import io
from tqdm.auto import tqdm


def main(from_folder, to_folder, min_volume=23442, min_pix_sum=3263,
         min_ratio=1/3, max_ratio=3):
    image_stats = []

    for p in tqdm(list(from_folder.glob("*/*.tif"))):
        image = io.imread(p, as_gray=True)
        h, w = image.shape
        stat_dict = {"path": p, "pix_sum": image.sum(),
                     "h": h, "w": w}
        image_stats.append(stat_dict)

    image_stats_df = pd.DataFrame(image_stats)
    image_stats_df["ratio"] = image_stats_df.h / image_stats_df.w
    image_stats_df["volume"] = image_stats_df.h * image_stats_df.w

    good_slice = image_stats_df[(image_stats_df.volume > min_volume)
                                & (image_stats_df.pix_sum > min_pix_sum)
                                & (image_stats_df.ratio > min_ratio)
                                & (image_stats_df.ratio < max_ratio)]
    for p in tqdm(good_slice.path):
        to_path = to_folder / p.relative_to(from_folder)
        to_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(str(p), str(to_path))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--from_folder', type=Path, default=Path('../data/CleanProcessed'))
    parser.add_argument('--to_folder', type=Path, default=Path('../data/FilteredCleanProcessed'))

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.from_folder, args.to_folder)
