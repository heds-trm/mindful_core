import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from mindful_core.utils.data_constants import SCAN_ID, SUBSET_ID, LABEL
from mindful_core.utils.misc import try_load_json


# region Load
def load_test_inferences_from_path(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)

    if not path.exists():
        # raise FileNotFoundError(path)
        print(f"Warning {path} does not exist")
        return None

    if path.is_dir():
        path = path / "test_inferences.csv"

    if not path.exists():
        folded_test_inferences = load_folded_test_inferences(path.parent)
        if folded_test_inferences is None:
            # raise FileNotFoundError(path)
            print(f"Warning {path} does not exist")
            return None
        else:
            folded_test_inferences.to_csv(path)

    if path.suffix == ".xlsx":
        # noinspection PyTypeChecker
        data_frame = pd.read_excel(path, index_col=SCAN_ID)
    else:
        data_frame = pd.read_csv(path, index_col=SCAN_ID)

    return data_frame


def load_folded_test_inferences(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)

    if path.is_file():
        path = path.parent

    inferences_paths = list(path.rglob("inferences_fold_*.csv"))
    if len(inferences_paths) == 0:
        return None

    test_inferences_disjoint = []
    existing_ids = []
    for inferences_path in inferences_paths:
        inferences = pd.read_csv(inferences_path, index_col=SCAN_ID)

        # region Only keep test samples
        if SUBSET_ID in inferences.columns:
            is_test = inferences[SUBSET_ID] == "test"
            inferences = inferences[is_test]
        # endregion

        # region Check for IDs that may already be accounted for
        new_ids = inferences.index.to_list()
        ids_present_twice = [new_id for new_id in new_ids if new_id in existing_ids]
        if len(ids_present_twice) > 0:
            raise KeyError("IDs `{}` are present at least twice in separate test folds in {}".
                           format(ids_present_twice, inferences_path))
        existing_ids += new_ids
        # endregion

        test_inferences_disjoint.append(inferences)

    test_inferences = pd.concat(test_inferences_disjoint)
    return test_inferences


def load_test_inferences_from_config(config: str | Path | dict[str, Any]) -> dict[str, pd.DataFrame]:
    if not isinstance(config, dict):
        config = try_load_json(config, "Test Inferences config")

    data = {exp_id: load_test_inferences_from_path(path)
            for exp_id, path in config.items()
            if not exp_id.startswith("-")}

    return {exp_id: test_inferences
            for exp_id, test_inferences in data.items()
            if test_inferences is not None}


def load_test_inferences_from_config_dict(config: dict[str, Any],
                                          log_dir: str,
                                          skip_exp: bool
                                          ) -> dict[str, pd.DataFrame]:
    all_test_inferences = {}
    for exp_id in config:
        if not exp_id.startswith("-"):
            if skip_exp and "skip" in config[exp_id] and config[exp_id]["skip"] == "yes":
                test_inferences = None
            else:
                test_inferences = load_test_inferences_from_path(Path(log_dir) / Path(exp_id))
            if test_inferences is not None:
                all_test_inferences[exp_id] = test_inferences
    return all_test_inferences


def get_experiment_test_inferences(experiment_path: str,
                                   test_partitions: list[pd.Index]
                                   ):
    test_inferences = pd.read_csv(experiment_path, index_col=SCAN_ID)
    return [test_inferences.loc[test_partition] for test_partition in test_partitions]


# endregion


# region Ground truth / Probabilities extraction

def extract_ground_truth_and_probabilities(test_inferences: dict[str, pd.DataFrame],
                                           folds_ids: list[pd.Index]
                                           ) -> tuple[list[np.ndarray] | None, dict[str, list[np.ndarray]]]:
    ground_truth = None
    for exp_id in test_inferences:
        exp_inferences = test_inferences[exp_id]
        exp_inferences = [exp_inferences.loc[fold_ids] for fold_ids in folds_ids]
        if ground_truth is None:
            ground_truth = [get_ground_truth(fold_inferences) for fold_inferences in exp_inferences]
        # noinspection PyTypeChecker
        test_inferences[exp_id] = [get_probabilities(fold_inferences) for fold_inferences in exp_inferences]
    # noinspection PyTypeChecker
    return ground_truth, test_inferences


def ground_truths_and_probabilities_for_datasets(test_inferences: dict[str, pd.DataFrame],
                                                 datasets: dict[str, Any]
                                                 ) -> tuple[list[np.ndarray] | None, dict[str, list[np.ndarray]]]:
    ground_truths = {}
    for exp_id in test_inferences:
        exp_inferences = test_inferences[exp_id]
        dataset = datasets[exp_id]
        folds_ids = get_test_partitions(dataset["folds"])
        exp_inferences = [exp_inferences.loc[fold_ids] for fold_ids in folds_ids]
        ground_truth = [get_ground_truth(fold_inferences) for fold_inferences in exp_inferences]
        ground_truths[exp_id] = ground_truth
        # noinspection PyTypeChecker
        test_inferences[exp_id] = [get_probabilities(fold_inferences) for fold_inferences in exp_inferences]
    # noinspection PyTypeChecker
    return ground_truths, test_inferences


def get_ground_truth(experiment_inferences: pd.DataFrame) -> np.ndarray:
    return np.asarray(experiment_inferences[LABEL], dtype=np.int32)


def get_probabilities(experiment_inferences: pd.DataFrame) -> np.ndarray:
    return np.asarray(experiment_inferences["Probability"], dtype=np.float32)


# endregion

def get_test_partitions(folds_folder: str) -> list[pd.Index]:
    folds_folder: Path = Path(folds_folder)
    test_partitions = {}
    for filepath in folds_folder.iterdir():
        if filepath.match("*fold_*.csv"):
            index = int(filepath.stem.split("_")[-1])
            fold = pd.read_csv(filepath, index_col=SCAN_ID)
            fold = fold[fold[SUBSET_ID] == "test"]
            test_partitions[index] = fold.index

    test_partitions = [test_partitions[i] for i in range(len(test_partitions))]
    return test_partitions


def get_dataset_for_experiments(experiments: dict[str, Any], datasets: dict[dict, Any]) -> dict[str, Path]:
    experiment_datasets = {}
    for experiment_id in experiments:
        if ("dataset" in experiments[experiment_id]) and (experiments[experiment_id]["dataset"] in datasets):
            experiment_datasets[experiment_id] = datasets[experiments[experiment_id]["dataset"]]
    return experiment_datasets
