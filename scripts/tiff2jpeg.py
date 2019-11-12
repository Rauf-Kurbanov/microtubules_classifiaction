import argparse
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_path", type=Path)
    parser.add_argument("--jpeg_path", type=Path)

    return parser


def main(args):
    in_path, out_path = args.tiff_path, args.jpeg_path

    if in_path.is_file():
        im = Image.open(in_path)
        im.save(out_path, "JPEG")
    elif in_path.is_dir():
        tif_files = [f for f in in_path.rglob("*") if f.suffix in [".tif", ".tiff"]]
        for file in tqdm(tif_files):
            im = Image.open(file)
            rel_path = file.relative_to(in_path)
            name = str(rel_path).rstrip(file.suffix)
            save_path = out_path / f"{name}.jpg"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(save_path, "JPEG")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_known_args()[0]
    main(args)
