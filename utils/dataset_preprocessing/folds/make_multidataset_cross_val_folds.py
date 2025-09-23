import argparse
import os
from typing import Optional

from utils.parsing import safe_int
from data.data_folds.preset_fold import PresetFold
from utils.dataset_preprocessing.folds.make_cross_validation_folds import (
    make_multi_part_cross_validation_folds,
    save_folds)
from utils.dataset_preprocessing.folds.join_folds import load_and_join_additional_features


def add_dataset_name_to_scan_id(fold: PresetFold, dataset_name: str) -> PresetFold:
    for subset_id in fold.samples:
        for i in range(len(fold.samples[subset_id])):
            fold.samples[subset_id][i].id = "{}_{}".format(dataset_name, fold.samples[subset_id][i].id)
    return fold


def load_partitions(fold_paths: list[str], dataset_names: list[str], seed: int):
    return [add_dataset_name_to_scan_id(PresetFold(fold_path=fold_path, seed=seed), dataset_name)
            for fold_path, dataset_name in zip(fold_paths, dataset_names)]


def main():
    # region Arg parsing
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--fold_paths", nargs="+", required=True)
    arg_parser.add_argument("--additional_features_paths", nargs="+", required=True)
    arg_parser.add_argument("--dataset_names", nargs="+", required=True)

    arg_parser.add_argument("--folds_count", default=5)
    arg_parser.add_argument("--test_partitions_count", default=1)
    arg_parser.add_argument("--validation_partitions_count", default=1)
    arg_parser.add_argument("--output_type", default="csv", choices=["both", "csv", "json"])
    arg_parser.add_argument("--seed", default=3930787959)

    args = arg_parser.parse_args()
    fold_paths: list[str] = args.fold_paths
    features_paths: list[str] = args.additional_features_paths
    dataset_names: list[str] = args.dataset_names

    folds_count: int = safe_int(args.folds_count, default=5)
    test_partitions_count: int = safe_int(args.test_partitions_count, default=1)
    validation_partitions_count: int = safe_int(args.validation_partitions_count, default=1)
    output_type: str = args.output_type
    seed: Optional[int] = safe_int(args.seed, default=None)
    # endregion

    if (len(fold_paths) != len(features_paths)) or (len(fold_paths) != len(dataset_names)):
        raise ValueError

    # region Main folds files
    partitions = load_partitions(fold_paths, dataset_names, seed)

    folds = make_multi_part_cross_validation_folds(partitions,
                                                   folds_count=folds_count,
                                                   test_partitions_count=test_partitions_count,
                                                   validation_partitions_count=validation_partitions_count)
    partition_common_path = os.path.commonpath(fold_paths)
    joint_dataset_name = "_".join(dataset_names)
    save_folds(folds, partition_common_path, joint_dataset_name, output_type)
    # endregion

    # region Additional features
    features = load_and_join_additional_features(features_paths, dataset_names)
    features_filename = "_".join(dataset_names) + "_joint_features.csv"
    features_common_path = os.path.commonpath(features_paths)
    features_output_path = os.path.join(features_common_path, features_filename)
    features.to_csv(features_output_path, index=False)
    # endregion


if __name__ == "__main__":
    main()
