import pandas as pd
from pathlib import Path
import argparse

from mindful_core.utils.data_constants import SCAN_ID, DEFAULT_IMAGE_COLUMN


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--inputs", nargs="+")
    arg_parser.add_argument("--output")
    args = arg_parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, Path] = {}
    for i in range(0, len(args.inputs), 2):
        key = "image:{}".format(args.inputs[i])
        inputs[key] = Path(args.inputs[i + 1])

    additional_folds: dict[int, dict[str, pd.DataFrame]] = {}
    output_folds: dict[int, pd.DataFrame] = {}
    for key, root in inputs.items():
        for fold_path in root.glob("*fold_*.csv"):
            fold_id = int(fold_path.stem.split("_")[-1])
            fold = pd.read_csv(fold_path, index_col=SCAN_ID)
            image_column = DEFAULT_IMAGE_COLUMN
            fold = fold.rename(columns={image_column: key})

            if fold_id not in additional_folds:
                additional_folds[fold_id] = {}
                output_folds[fold_id] = fold
            else:
                additional_folds[fold_id][key] = fold

    for fold_id, output_fold in output_folds.items():
        for key, additional_fold in additional_folds[fold_id].items():
            output_fold[key] = additional_fold[key]
        output_fold.to_csv(output / "fold_{:02d}.csv".format(fold_id))


if __name__ == "__main__":
    main()
