from pathlib import Path
import pandas as pd
from modules.data import get_data  # TODO deprecated (Rauf 20.05.20)
from modules.utils import Mode


def preprocess(labels_df, data_root):
    labels_df.filepath = labels_df.filepath.apply(
        lambda fp: str(Path(fp).relative_to(data_root)))

    replace_dict = {"01Taxol": 0, "1Taxol": 1, "Control": 2}
    labels_df["label"] = labels_df["class"].apply(lambda c: replace_dict[c])
    labels_df = labels_df.sample(frac=1).reset_index()

    return labels_df


def split_df(data_dir, seed, each_class=100):
    labels_df, class_names, num_classes = get_data(data_dir, mode=Mode.ZERO_VS_ZERO_ONE_VS_ONE)
    data_root = data_dir.parent
    labels_df = preprocess(labels_df, data_root)
    test_parts = []
    for cn in class_names:
        part = labels_df[labels_df["class"] == cn].sample(each_class, random_state=seed)
        test_parts.append(part)
    test_data = pd.concat(test_parts)
    test_data = test_data.sample(frac=1., random_state=seed)
    train_data = labels_df[~labels_df.index.isin(test_data.index)]

    return labels_df, train_data, test_data


def main():
    DATA_ROOT = Path("/Users/raufkurbanov/Programs/microtubules_classifiaction/data")
    SPLIT_ROOT = Path("/Users/raufkurbanov/Programs/microtubules_classifiaction/data/splits")
    SEED = 42

    for data_dir in ["FilteredCleanProcessed"]:
        labels_df, train_data, test_data = split_df(DATA_ROOT / data_dir, SEED)
        save_dir = SPLIT_ROOT / data_dir

        save_dir.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(save_dir / "labels.csv")
        train_data.to_csv(save_dir / "train.csv")
        test_data.to_csv(save_dir / "test.csv")


if __name__ == '__main__':
    main()
