import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os
import argparse
from typing import Any

from mindful_core.utils.data_constants import SCAN_ID, SUBSET_ID, LABEL
from mindful_core.utils.misc import try_load_json
from mindful_core.analysis.statistics.roc_compare import compute_roc_curve, compute_kfolds_roc_distribution
from mindful_core.analysis.statistics.classification_summary import compute_auroc


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


def load_test_inferences_from_path(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        path = path / "test_inferences.csv"

    if not path.exists():
        folded_test_inferences = load_folded_test_inferences(path.parent)
        if folded_test_inferences is None:
            raise FileNotFoundError(path)
        else:
            folded_test_inferences.to_csv(path)

    if path.suffix == ".xlsx":
        # noinspection PyTypeChecker
        data_frame = pd.read_excel(path, index_col=SCAN_ID)
    else:
        data_frame = pd.read_csv(path, index_col=SCAN_ID)

    return data_frame


def load_test_inferences_from_config(config: str | Path | dict[str, Any]) -> dict[str, pd.DataFrame]:
    if not isinstance(config, dict):
        config = try_load_json(config, "Test Inferences config")

    return {exp_id: load_test_inferences_from_path(path)
            for exp_id, path in config.items()
            if not exp_id.startswith("-")}


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


def get_experiment_test_inferences(experiment_path: str,
                                   test_partitions: list[pd.Index]
                                   ):
    test_inferences = pd.read_csv(experiment_path, index_col=SCAN_ID)
    return [test_inferences.loc[test_partition] for test_partition in test_partitions]


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


def get_ground_truth(experiment_inferences: pd.DataFrame) -> np.ndarray:
    return np.asarray(experiment_inferences[LABEL], dtype=np.int32)


def get_probabilities(experiment_inferences: pd.DataFrame) -> np.ndarray:
    return np.asarray(experiment_inferences["Probability"], dtype=np.float32)


class KFoldROCSummary(object):
    def __init__(self,
                 ground_truth: list[np.ndarray],
                 probabilities: list[np.ndarray],
                 name: str,
                 n_thresholds: int = 1000,
                 color: str = None,
                 bootstrapped_auroc: str = None,
                 ):
        self.ground_truth = ground_truth
        self.probabilities = probabilities
        self.name = name
        self.n_thresholds = n_thresholds
        self.color = color
        self.bootstrapped_auroc = bootstrapped_auroc

        self.roc_curves = compute_roc_curve(ground_truth, probabilities, n_thresholds=n_thresholds)
        if len(self.roc_curves) > 1:
            (fpr_mean, tpr_mean), (_, tpr_std) = compute_kfolds_roc_distribution(self.roc_curves)
        else:
            fpr_mean, tpr_mean = self.roc_curves[0][0].mean(), self.roc_curves[0][1].mean()
            tpr_std = None

        self.fpr_mean = fpr_mean
        self.tpr_mean = tpr_mean

        if tpr_std is not None:
            self.tpr_upper = np.minimum(tpr_mean + tpr_std, 1.0)
            self.tpr_lower = np.maximum(tpr_mean - tpr_std, 0.0)
        else:
            self.tpr_upper = self.tpr_lower = None

        # using same method as with ClassificationSummary
        aurocs = [compute_auroc(torch.as_tensor(_probabilities), torch.as_tensor(_labels)).numpy() * 100.0
                  for (_labels, _probabilities) in zip(ground_truth, probabilities)]
        # aurocs = [auc(fpr, tpr) * 100.0 for (fpr, tpr) in self.roc_curves]
        self.auroc_mean, self.auroc_std = round(np.mean(aurocs), 1), round(np.std(aurocs), 1)

    def plot_std_area(self, axis: plt.Axes, alpha=0.2) -> None:
        if (self.tpr_lower is None) or (self.tpr_upper is None):
            return
        
        axis.fill_between(self.fpr_mean, self.tpr_lower, self.tpr_upper, alpha=alpha, color=self.color)

    def plot_std_bounds(self, axis: plt.Axes, alpha=0.2) -> None:
        if (self.tpr_lower is None) or (self.tpr_upper is None):
            return

        axis.plot(self.fpr_mean, self.tpr_lower, alpha=alpha, color=self.color)
        axis.plot(self.fpr_mean, self.tpr_upper, alpha=alpha, color=self.color)

    def plot_mean(self, axis: plt.Axes, alpha=1.0) -> None:
        axis.plot(self.fpr_mean, self.tpr_mean, label=self.label, color=self.color, alpha=alpha)

    @property
    def label(self) -> str:
        if self.bootstrapped_auroc is None:
            auroc_label = "{} (+/- {})".format(self.auroc_mean, self.auroc_std)
        else:
            auroc_label = self.bootstrapped_auroc
        label = "{} - AUROC: {}".format(self.name, auroc_label)
        return label


def draw_roc_comparisons(config_path: Path, 
                         n_thresholds: int = 1000, 
                         sensitivity_thresholds: list[float] = None
                         ) -> Path:
    sensitivity_thresholds = sensitivity_thresholds or []

    # region Load
    config = try_load_json(config_path, "ROC Comparison config")
    folds_ids = get_test_partitions(config["folds"])
    test_inferences = load_test_inferences_from_config(config["experiments"])
    bootstrapped_aurocs = config.get("bootstrapped_aurocs", {})
    # endregion

    figure, axis = plt.subplots()
    color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                   "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    ground_truth, test_probabilities = extract_ground_truth_and_probabilities(test_inferences, folds_ids)
    summaries = [KFoldROCSummary(ground_truth, exp_probabilities, exp_id, n_thresholds,
                                 color=color_cycle[i % len(color_cycle)],
                                 bootstrapped_auroc=bootstrapped_aurocs.get(exp_id))
                 for i, (exp_id, exp_probabilities) in enumerate(test_probabilities.items())]

    for summary in summaries:
        summary.plot_std_area(axis, alpha=0.1)

    for summary in summaries:
        summary.plot_std_bounds(axis, alpha=0.025)

    for summary in summaries:
        summary.plot_mean(axis, alpha=1.0)

    for sensitivity_threshold in sensitivity_thresholds:
        axis.plot([0.0, 1.0],
                  [sensitivity_threshold, sensitivity_threshold],
                  color="red", linestyle="dashed", alpha=0.5,
                  label="{}% sensitivity".format(round(sensitivity_threshold * 100)))

    axis.legend()

    save_filepath = config_path.parent / "{}.png".format(config_path.stem)
    figure.savefig(save_filepath)

    return save_filepath


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_paths", type=str, nargs="+")
    arg_parser.add_argument("--n_thresholds", type=int, default=1000)
    arg_parser.add_argument("--sensitivity_thresholds", nargs="+")
    args = arg_parser.parse_args()

    config_paths = [Path(path) for path in args.config_paths]
    n_thresholds = int(args.n_thresholds)
    sensitivity_thresholds = [float(sensitivity_threshold) for sensitivity_threshold in args.sensitivity_thresholds]

    output_paths: list[Path] = []
    for config_path in config_paths:
        output_path = draw_roc_comparisons(config_path, n_thresholds, sensitivity_thresholds)
        output_paths.append(output_path)

    if len(output_paths) > 1:
        images = [cv2.imread(path.as_posix()) for path in output_paths]
        joint_image = cv2.hconcat(images)
        root = os.path.commonpath(output_paths)
        joint_image_path = Path(root, "joint_roc_comparison.png")
        cv2.imwrite(joint_image_path.as_posix(), joint_image)

if __name__ == "__main__":
    main()
