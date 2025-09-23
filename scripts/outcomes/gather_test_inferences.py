import pandas as pd
from pathlib import Path
import argparse


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("root", type=str)

    args = arg_parser.parse_args()
    root = Path(args.root)

    test_inferences_filepaths: list[Path] = []
    for filepath in root.glob("**/test_inferences.csv"):
        if not filepath.parent.match("*/version_*"):
            test_inferences_filepaths.append(filepath)

    compilation = pd.DataFrame()
    error_count, entry_count = None, None
    for filepath in test_inferences_filepaths:
        test_inference = pd.read_csv(filepath, index_col="ScanID")
        test_id = filepath.relative_to(root).parent

        correct_predictions = test_inference["Correct Prediction"]
        errors = (~correct_predictions).astype(int)
        entries = errors.replace(0, 1)
        if error_count is None:
            error_count = errors
            entry_count = entries
            compilation["Label"] = test_inference["Label"]
        else:
            error_count += errors
            entry_count += entries
        compilation[test_id] = correct_predictions

    compilation["Error Ratio"] = error_count / entry_count
    compilation = compilation.sort_values(by="Error Ratio", ascending=False)

    compilation.to_csv(root / "test_inferences_compilation.csv")


if __name__ == "__main__":
    main()
