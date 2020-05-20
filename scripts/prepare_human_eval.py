from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def main():
    DATA_ROOT = Path("/Users/raufkurbanov/Programs/microtubules_classifiaction/data")
    SPLIT_ROOT = Path("/Users/raufkurbanov/Programs/microtubules_classifiaction/data/splits")
    HUMAN_JPG_ROOT = DATA_ROOT / "human_jpg"

    for data_dir in ["FilteredCleanProcessed"]:
        print("data_dir", data_dir)

        train_df = pd.read_csv(SPLIT_ROOT / data_dir / "train.csv", index_col=0).reset_index()
        test_df = pd.read_csv(SPLIT_ROOT / data_dir / "test.csv", index_col=0).reset_index()

        for split, df in [("train", train_df), ("test", test_df)]:
            print("split", split)
            save_dir = HUMAN_JPG_ROOT / data_dir / split
            save_dir.mkdir(parents=True, exist_ok=True)

            for index, row in tqdm(df.iterrows(), total=len(df)):
                tif_path = DATA_ROOT / row['filepath']
                jpg_path = save_dir / f"{row['index']}.jpg"

                im = Image.open(tif_path)
                im.save(jpg_path, "JPEG")


if __name__ == '__main__':
    main()
