import pandas as pd
from pathlib import Path
import argparse


def filepath_to_partition_id(filepath: Path) -> int:
    filename = filepath.stem
    *_, partition_id = filename.split("_")
    return int(partition_id)


def folds_to_partitions(root: Path) -> pd.DataFrame:
    partitions_data = {"PartitionID": {}}
    columns = None

    for filepath in root.glob(pattern="*fold*.csv"):
        fold = pd.read_csv(filepath, index_col="ScanID")
        subset_ids = fold.pop("SubsetID")

        if columns is None:
            columns = fold.columns
            for column_name in columns:
                partitions_data[column_name] = {}

        partition_id = filepath_to_partition_id(filepath)
        for scan_id in fold.index:
            if subset_ids[scan_id] != "test":
                continue

            partitions_data["PartitionID"][scan_id] = partition_id
            for column_name in columns:
                partitions_data[column_name][scan_id] = fold[column_name][scan_id]

    partitions = pd.DataFrame(partitions_data)
    partitions.index.name = "ScanID"
    return partitions


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("root")
    args = arg_parser.parse_args()
    root = Path(args.root)

    partitions = folds_to_partitions(root)
    partitions.to_csv(root / "partitions.csv")


if __name__ == "__main__":
    main()
