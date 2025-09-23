import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from analysis.statistics.roc_compare import compute_roc_curve, compute_kfolds_roc_distribution
from analysis.statistics.classification_summary import compute_auroc
from scripts.outcomes.compare_experiments_delong import load_test_inferences_from_config
from utils.misc import load_json


def get_test_partitions(folds_folder: str) -> list[pd.Index]:
    folds_folder: Path = Path(folds_folder)
    test_partitions = {}
    for filepath in folds_folder.iterdir():
        if filepath.match("*_fold_*.csv"):
            index = int(filepath.stem.split("_")[-1])
            fold = pd.read_csv(filepath, index_col="ScanID")
            fold = fold[fold["SubsetID"] == "test"]
            test_partitions[index] = fold.index

    test_partitions = [test_partitions[i] for i in range(len(test_partitions))]
    return test_partitions


def get_experiment_test_inferences(experiment_path: str,
                                   test_partitions: list[pd.Index]
                                   ):
    test_inferences = pd.read_csv(experiment_path, index_col="ScanID")
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
    return np.asarray(experiment_inferences["Label"], dtype=np.int32)


def get_probabilities(experiment_inferences: pd.DataFrame) -> np.ndarray:
    return np.asarray(experiment_inferences["Probability"], dtype=np.float32)


class KFoldROCSummary(object):
    def __init__(self,
                 ground_truth: list[np.ndarray],
                 probabilities: list[np.ndarray],
                 name: str,
                 n_thresholds: int = 1000,
                 color: str = None,
                 ):
        self.ground_truth = ground_truth
        self.probabilities = probabilities
        self.name = name
        self.n_thresholds = n_thresholds
        self.color = color

        self.roc_curves = compute_roc_curve(ground_truth, probabilities, n_thresholds=n_thresholds)
        (fpr_mean, tpr_mean), (_, tpr_std) = compute_kfolds_roc_distribution(self.roc_curves)

        self.fpr_mean = fpr_mean
        self.tpr_mean = tpr_mean
        self.tpr_upper = np.minimum(tpr_mean + tpr_std, 1.0)
        self.tpr_lower = np.maximum(tpr_mean - tpr_std, 0.0)

        # using same method as with ClassificationSummary
        aurocs = [compute_auroc(torch.as_tensor(_probabilities), torch.as_tensor(_labels)).numpy() * 100.0
                  for (_labels, _probabilities) in zip(ground_truth, probabilities)]
        # aurocs = [auc(fpr, tpr) * 100.0 for (fpr, tpr) in self.roc_curves]
        self.auroc_mean, self.auroc_std = round(np.mean(aurocs), 1), round(np.std(aurocs), 1)

    def plot_std_area(self, axis: plt.Axes, alpha=0.2) -> None:
        axis.fill_between(self.fpr_mean, self.tpr_lower, self.tpr_upper, alpha=alpha, color=self.color)

    def plot_std_bounds(self, axis: plt.Axes, alpha=0.2) -> None:
        axis.plot(self.fpr_mean, self.tpr_lower, alpha=alpha, color=self.color)
        axis.plot(self.fpr_mean, self.tpr_upper, alpha=alpha, color=self.color)

    def plot_mean(self, axis: plt.Axes, alpha=1.0) -> None:
        label = "{} - AUROC: {} (+/- {})".format(self.name, self.auroc_mean, self.auroc_std)
        axis.plot(self.fpr_mean, self.tpr_mean, label=label, color=self.color, alpha=alpha)


def main():
    # region Arg parsing
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_path", type=str)
    arg_parser.add_argument("--n_thresholds", type=int, default=1000)
    arg_parser.add_argument("--sensitivity_thresholds", nargs="+")
    args = arg_parser.parse_args()

    n_thresholds = int(args.n_thresholds)
    config_path = Path(args.config_path)
    sensitivity_thresholds = [float(sensitivity_threshold) for sensitivity_threshold in args.sensitivity_thresholds]
    # endregion

    # region Load
    config = load_json(config_path)
    folds_ids = get_test_partitions(config["folds"])
    test_inferences = load_test_inferences_from_config(config["experiments"])
    # endregion

    figure, axis = plt.subplots()
    color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                   "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    ground_truth, test_probabilities = extract_ground_truth_and_probabilities(test_inferences, folds_ids)
    summaries = [KFoldROCSummary(ground_truth, exp_probabilities, exp_id, n_thresholds,
                                 color=color_cycle[i % len(color_cycle)])
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

    save_filepath = config_path.parent / "roc_comparisons.png"
    figure.savefig(save_filepath)


if __name__ == "__main__":
    main()
