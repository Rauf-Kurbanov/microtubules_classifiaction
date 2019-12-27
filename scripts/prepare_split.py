from pathlib import Path
from catalyst.utils.dataset import split_dataframe
from modules.data import get_data


def preprocess(labels_df, data_root):
    labels_df.filepath = labels_df.filepath.apply(
        lambda fp: str(Path(fp).relative_to(data_root)))

    replace_dict = {"Control": "0mkg", "control": "0mkg", "01Taxol": "01mkg",
                    "1Taxol": "1mkg"}
    labels_df["class"] = labels_df["class"].apply(lambda c:
                                                  replace_dict[
                                                      c] if c in replace_dict else c)
    labels_df = labels_df[labels_df["class"] != "5mkg"]

    replace_dict = {"0mkg": 0, "01mkg": 1, "1mkg": 2}
    labels_df["label"] = labels_df["class"].apply(lambda c: replace_dict[c])
    labels_df = labels_df.sample(frac=1).reset_index()

    return labels_df


def split_df(data_dir, seed, test_size=300):
    labels_df, class_names, num_classes = get_data(data_dir)
    data_root = data_dir.parent
    labels_df = preprocess(labels_df, data_root)
    train_data, test_data = split_dataframe(labels_df, test_size=test_size,
                                            random_state=seed)

    return labels_df, train_data, test_data


def main():
    DATA_ROOT = Path("/Users/raufkurbanov/Data/microtubules")
    SPLIT_ROOT = Path("/Users/raufkurbanov/Programs/microtubules_classifiaction/data/splits")
    SEED = 42

    for data_dir in ["NewArchive", "OldArchive"]:
        labels_df, train_data, test_data = split_df(DATA_ROOT / data_dir, SEED)
        save_dir = SPLIT_ROOT / data_dir

        save_dir.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(save_dir / "labels.csv")
        train_data.to_csv(save_dir / "train.csv")
        test_data.to_csv(save_dir / "test.csv")


if __name__ == '__main__':
    main()
