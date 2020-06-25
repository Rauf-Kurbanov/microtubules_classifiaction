from pathlib import Path

import pandas as pd
import ttach as tta
from catalyst.dl import SupervisedRunner
from sklearn.metrics import f1_score, accuracy_score

from modules.data import filter_data_by_mode, get_test_loader
from modules.models import ClassificationNet, LargerClassificationNet
from modules.utils import get_nn_parser


def main(config):
    test_df = pd.read_csv(config.split_path / "test.csv",
                          usecols=["class", "filepath", "label"])
    test_df.filepath = test_df.filepath.apply(lambda p: config.data_root / p)
    test_df, class_names, num_classes = filter_data_by_mode(test_df, config.mode)

    if config.debug:
        test_df = test_df[:config.batch_size]

    test_data = test_df.to_dict('records')
    test_loader = get_test_loader(data_dir=config.data_root / config.data, test_data=test_data,
                                  num_classes=num_classes)

    model_fn = {1: ClassificationNet,
                2: LargerClassificationNet}[config.n_layers]

    model = model_fn(backbone_name=config.backbone,
                     n_classes=num_classes)
    if config.tta:
        model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform())

    runner = SupervisedRunner(device=config.device)
    ckpth_path = Path(config.config).parent / "checkpoints" / "best.pth"
    output = runner.predict_loader(model, loader=test_loader,
                                   resume=ckpth_path,
                                   verbose=True)

    pred = output.argmax(1)
    gt = test_df.label.to_numpy()

    fscore = f1_score(gt, pred, average='macro')
    acc = accuracy_score(gt, pred)
    # f'{a:.2f}'
    print("fscore", f"{fscore:.4f}")
    print("acc", f"{acc:.4f}")


if __name__ == '__main__':
    parser = get_nn_parser()  # TODO other args, just pass results folder
    args, _ = parser.parse_known_args()
    assert args.fixed_split  # TODO

    main(args)
