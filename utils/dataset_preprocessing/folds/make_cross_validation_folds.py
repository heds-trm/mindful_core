import argparse
from pathlib import Path
from typing import Optional, Sequence

from mindful_core.data.data_folds import DataFold, PresetFold
from mindful_core.utils.parsing import safe_int


def join_sub_folds(sub_folds: list[list[DataFold]]) -> list[DataFold]:
    """

    ### Parameters
        1. sub_folds : list[list[DataFold]]
    """
    folds_count = len(sub_folds[0])
    return [DataFold.join_folds([sub_folds[j][i] for j in range(len(sub_folds))]) for i in range(folds_count)]


def make_multi_part_cross_validation_folds(collections: Sequence[DataFold],
                                           folds_count: int,
                                           test_partitions_count: int,
                                           validation_partitions_count: int
                                           ) -> list[DataFold]:
    """Distribute each collection's samples into C=`folds_count` sub-folds, yielding NxC sub-folds.
    Then, make C folds by concatenating the subsets of their N sub-folds.
    
    For example, with the two following collections:
    - [0, 1, 2, 3]
    - [A, B, C]

    When split into C=3 folds:
    - test : [0, 1] + [A], val : [2] + [B], train [3] + [C]
    - train : [0, 1] + [A], test : [2] + [B], val [3] + [C]
    - val : [0, 1] + [A], train : [2] + [B], test [3] + [C]

    ### Parameters
    1. collections
            - A sequence of collections (of samples).
    2. folds_count
            - `folds_count` : the number of folds to generate
    3. test_partitions_count
            - The number of partitions in a fold reserved for the test subset
    4. validation_partitions_count
            - The number of partitions in a fold reserved for the validation subset

    ### Returns
    - `folds_count` folds
    """
    sub_folds = [collection.make_cross_validation_folds(folds_count=folds_count,
                                                        test_partitions_count=test_partitions_count,
                                                        validation_partitions_count=validation_partitions_count)
                 for collection in collections]
    collections = join_sub_folds(sub_folds)
    return collections


def save_folds(folds: list[DataFold], folder: str | Path, dataset_name: str, output_type: str) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for i, fold in enumerate(folds):
        fold_id = "{}_fold_{:02d}".format(dataset_name, i)
        if output_type in ["both", "csv"]:
            output_path = folder / (fold_id + ".csv")
            fold.save_scan_data(output_path)

        if output_type in ["both", "json"]:
            output_path = folder / (fold_id + ".json")
            fold.save_scan_data(output_path)


def main():
    # region Arg parse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", required=True,
                            help="Path to a CSV file describing an existing fold.")
    arg_parser.add_argument("--dataset_name", required=True,
                            help="The name the new dataset will be given in the form of `{name}_fold_{id}.*`.")
    arg_parser.add_argument("--output_folder", default=None,
                            help="The fold where to write the new dataset fold files. The folder must exist.")
    arg_parser.add_argument("--seed", default=3930787959,
                            help="Defaults to 3930787959.")
    arg_parser.add_argument("--folds_count", default=5,
                            help="The number of folds partition to create.")
    arg_parser.add_argument("--partitions", default=1,
                            help="The number of partitions per fold attributed to the test set.")
    arg_parser.add_argument("--validation_partitions_count", default=1,
                            help="The number of partitions per fold attributed to the validation set.")
    arg_parser.add_argument("--output_type", default="csv", choices=["both", "csv", "json"],
                            help="Controls the format which will be used to write folds. Defaults to CSV.")

    args = arg_parser.parse_args()
    source = Path(args.source)
    dataset_name: str = args.dataset_name
    output_folder: Path | None = Path(args.output_folder) if args.output_folder is not None else None
    seed: Optional[int] = safe_int(args.seed, default=None)
    folds_count: int = safe_int(args.folds_count, default=5)
    test_partitions_count: int = safe_int(args.partitions, default=1)
    validation_partitions_count: int = safe_int(args.validation_partitions_count, default=1)
    output_type: str = args.output_type
    # endregion

    base_fold = PresetFold(fold_path=source, seed=seed, use_abs_path=False)
    folds = base_fold.make_cross_validation_folds(folds_count=folds_count,
                                                  test_partitions_count=test_partitions_count,
                                                  validation_partitions_count=validation_partitions_count)

    if output_folder is None:
        output_folder = source
    output_folder.mkdir(parents=True, exist_ok=True)

    save_folds(folds, output_folder, dataset_name, output_type)


if __name__ == "__main__":
    main()
