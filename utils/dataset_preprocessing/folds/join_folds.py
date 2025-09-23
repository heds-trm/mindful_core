import pandas as pd
from pathlib import Path
import os
import itertools
import argparse


def add_dataset_name_to_scan_id(data_frame: pd.DataFrame, dataset_name: str, column_id: str = "ScanID"):
    def _update_scan_id(_scan_id: str) -> str:
        return "{}_{}".format(dataset_name, _scan_id)

    data_frame[column_id] = data_frame[column_id].apply(_update_scan_id)
    return data_frame


def add_dataset_rel_path(data_frame: pd.DataFrame, rel_path: str, column_id: str = "image:image"):
    def _update_filepath(_filepath: str) -> str:
        return os.path.join(rel_path, _filepath)

    data_frame[column_id] = data_frame[column_id].apply(_update_filepath)
    return data_frame


def load_and_join_additional_features(features_paths: list[str],
                                      dataset_names: list[str]
                                      ) -> pd.DataFrame:
    if len(dataset_names) != len(features_paths):
        raise ValueError("`features_paths` and `dataset_names` must have the same length, "
                         "got {} (fold paths) and {} (dataset names).".
                         format(len(features_paths), len(dataset_names)))

    features = [pd.read_csv(additional_features_path) for additional_features_path in features_paths]
    for i in range(len(features)):
        features[i] = add_dataset_name_to_scan_id(features[i], dataset_names[i])

    return pd.concat(features)


def load_fold(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def join_folds(fold_paths: list[Path],
               dataset_names: list[str],
               scalar_features_paths: list[Path] = None,
               categorical_features_paths: list[Path] = None,
               ) -> tuple[list[pd.DataFrame], pd.DataFrame | None, pd.DataFrame | None]:
    # region Sanity checks
    dataset_count = len(dataset_names)

    check_paths_and_count(fold_paths, dataset_count)
    check_paths_and_count(scalar_features_paths, dataset_count)
    check_paths_and_count(categorical_features_paths, dataset_count)
    # endregion

    fold_paths = match_fold_count(fold_paths)
    fold_count = len(fold_paths[0])

    folds = []
    for i in range(fold_count):
        i_th_folds = [load_fold(dataset_folds[i]) for dataset_folds in fold_paths]
        i_th_folds = [add_dataset_name_to_scan_id(dataset_fold, dataset_name)
                      for dataset_fold, dataset_name in zip(i_th_folds, dataset_names)]
        fold = pd.concat(i_th_folds)
        folds.append(fold)

    if scalar_features_paths is not None:
        scalar_features_paths = [path / "scalar_features.csv" if path.is_dir() else path
                                 for path in scalar_features_paths]
        scalar_features = pd.concat([add_dataset_name_to_scan_id(pd.read_csv(path), dataset_name)
                                     for path, dataset_name in zip(scalar_features_paths, dataset_names)])
    else:
        scalar_features = None

    if categorical_features_paths is not None:
        categorical_features_paths = [path / "categorical_features.csv" if path.is_dir() else path
                                      for path in categorical_features_paths]
        categorical_features = pd.concat([add_dataset_name_to_scan_id(pd.read_csv(path), dataset_name)
                                          for path, dataset_name in zip(categorical_features_paths, dataset_names)])
    else:
        categorical_features = None

    return folds, scalar_features, categorical_features


def optional_paths(paths: list[str] | None) -> list[Path] | None:
    if paths is None:
        return None
    return [Path(path) for path in paths]


def check_paths_and_count(paths: list[Path] | None, expected_count: int):
    if paths is None:
        return

    if len(paths) != expected_count:
        raise ValueError("Expected to find {} paths, but got {}.".
                         format(expected_count, len(paths)))
    missing_files = [path for path in paths if not path.exists()]

    if len(missing_files) > 0:
        raise FileNotFoundError("The following path(s) do not exist: {}".format(missing_files))


def match_fold_count(fold_paths: list[Path]) -> list[list[Path]]:
    fold_paths = [list(sorted(path.glob("*fold_*.csv"))) if path.is_dir() else [path] for path in fold_paths]
    max_fold_count = max([len(paths) for paths in fold_paths])

    if max_fold_count > 1:
        for i in range(len(fold_paths)):
            if len(fold_paths[i]) == 1:
                fold_paths[i] = fold_paths[i] * max_fold_count
            elif len(fold_paths[i]) != max_fold_count:
                raise ValueError("When providing more than one fold file, you must provide dataset with the same "
                                 "number of fold. Max is {} and got {}.".
                                 format(max_fold_count, len(fold_paths[i])))

    return fold_paths


def save_joint_folds(output_dir: Path,
                     folds: list[pd.DataFrame],
                     scalar_features: pd.DataFrame | None = None,
                     categorical_features: pd.DataFrame | None = None
                     ) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, fold in enumerate(folds):
        fold.to_csv(output_dir / "fold_{:02d}.csv".format(i), index=False)

    if scalar_features is not None:
        scalar_features.to_csv(output_dir / "scalar_features.csv", index=False)

    if categorical_features is not None:
        categorical_features.to_csv(output_dir / "categorical_features.csv", index=False)


def get_optional_paths_list(keys: list[str], paths_dict: dict[str, Path] | None) -> list[Path] | None:
    if paths_dict is None:
        return None

    sub_list = [paths_dict[key] for key in keys if key in paths_dict]
    if len(sub_list) == 0:
        return None

    return sub_list


def make_multi_dataset_folds(output_dir: Path,
                             datasets_folds: dict[str, Path],
                             scalar_paths: dict[str, Path | None] | None,
                             categorical_paths: dict[str, Path | None] | None,
                             ) -> None:
    datasets = list(datasets_folds.keys())
    combinations = []
    for i in range(2, len(datasets) + 1):
        combinations += list(itertools.combinations(datasets, i))

    for combination_names in combinations:
        fold_paths = [datasets_folds[dataset] for dataset in combination_names]
        combination_scalar_paths = get_optional_paths_list(combination_names, scalar_paths)
        combination_categorical_paths = get_optional_paths_list(combination_names, categorical_paths)

        folds, scalar_features, categorical_features = join_folds(fold_paths,
                                                                  combination_names,
                                                                  combination_scalar_paths,
                                                                  combination_categorical_paths)

        combination_output_dir_name = "+".join(combination_names)
        combination_output_path = output_dir / combination_output_dir_name
        save_joint_folds(combination_output_path, folds, scalar_features, categorical_features)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("folds", nargs="+", help="Paths to either fold directories or CSV files.")
    arg_parser.add_argument("--output_dir",
                            help="The directory to which new files will be written. "
                                 "Will be created if it does not exist.")
    arg_parser.add_argument("--dataset_names", nargs="+", required=True,
                            help="The name to append before the ID of each sample")
    arg_parser.add_argument("--scalar_features", nargs="+", default=None,
                            help="Paths to scalar feature CSV files. "
                                 "If directories are provided, assumes files to be called `scalar_features.csv`")
    arg_parser.add_argument("--categorical_features", nargs="+", default=None,
                            help="Paths to categorical feature CSV files. "
                                 "If directories are provided, assumes files to be called `categorical_features.csv`")

    args = arg_parser.parse_args()
    fold_paths: list[Path] = [Path(path) for path in args.folds]
    output_dir: Path = Path(args.output_dir)
    dataset_names: list[str] = args.dataset_names
    scalar_features_paths = optional_paths(args.scalar_features)
    categorical_features_paths = optional_paths(args.categorical_features)

    folds, scalar_features, categorical_features = join_folds(fold_paths,
                                                              dataset_names,
                                                              scalar_features_paths,
                                                              categorical_features_paths)
    save_joint_folds(output_dir, folds, scalar_features, categorical_features)


if __name__ == "__main__":
    main()
