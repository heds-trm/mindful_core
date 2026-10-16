import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os
import argparse

from mindful_core.utils.misc import try_load_json
from mindful_core.analysis.statistics.roc_compare import compute_roc_curve, compute_kfolds_roc_distribution
from mindful_core.analysis.statistics.classification_summary import compute_auroc
from mindful_core.analysis.statistics.test_inferences import (get_dataset_for_experiments,
                                                              load_test_inferences_from_config_dict,
                                                              ground_truths_and_probabilities_for_datasets)


class KFoldROCSummary(object):
    def __init__(self,
                 ground_truth: list[np.ndarray],
                 probabilities: list[np.ndarray],
                 name: str,
                 n_thresholds: int = 1000,
                 color: str | None = None,
                 bootstrapped_auroc: str | None = None,
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

        aurocs = [compute_auroc(torch.as_tensor(_probabilities), torch.as_tensor(_labels)).numpy() * 100.0
                  for (_labels, _probabilities) in zip(ground_truth, probabilities)]
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

    def add_legend_outside(self, axis: plt.Axes, fontsize: int = 8) -> None:
        """
        Place legend outside the plot (right side) to avoid overlap.
        """
        axis.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=fontsize
        )

        # Ensure layout doesn't cut off legend
        # axis.figure.tight_layout()
        axis.figure.subplots_adjust(right=0.75)

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
                         sensitivity_thresholds: list[float] | None = None,
                         skip_exp=False
                         ) -> Path:
    sensitivity_thresholds = sensitivity_thresholds or []

    # region Load
    config = try_load_json(config_path, "ROC Comparison config")
    # folds_ids = get_test_partitions(config["folds"])
    # test_inferences = load_test_inferences_from_config(config["experiments"])
    datasets = get_dataset_for_experiments(config['experiments'], config["datasets"])
    log_dir = config["log_dir"]
    test_inferences = load_test_inferences_from_config_dict(config["experiments"], log_dir, skip_exp)

    bootstrapped_aurocs = config.get("bootstrapped_aurocs", {})
    # endregion

    figure, axis = plt.subplots()
    color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                   "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    ground_truths, test_probabilities = ground_truths_and_probabilities_for_datasets(test_inferences, datasets)

    summaries = []
    for i, (exp_id, exp_probabilities) in enumerate(test_probabilities.items()):
        summary = KFoldROCSummary(ground_truths[exp_id], exp_probabilities, exp_id, n_thresholds,
                                  color=color_cycle[i % len(color_cycle)],
                                  bootstrapped_auroc=bootstrapped_aurocs.get(exp_id))
        summaries.append(summary)
    # summaries = [KFoldROCSummary(ground_truth, exp_probabilities, exp_id, n_thresholds,
    #                             color=color_cycle[i % len(color_cycle)],
    #                             bootstrapped_auroc=bootstrapped_aurocs.get(exp_id))
    #             for i, (exp_id, exp_probabilities) in enumerate(test_probabilities.items())]

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

    # NEW: clean legend outside
    summaries[0].add_legend_outside(axis)

    # axis.legend()

    save_filepath = config_path.parent / "{}.png".format(config_path.stem)
    figure.savefig(save_filepath, bbox_inches="tight")

    return save_filepath


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_paths", type=str, nargs="+")
    arg_parser.add_argument("--n_thresholds", type=int, default=1000)
    arg_parser.add_argument("--sensitivity_thresholds", nargs="+")
    arg_parser.add_argument("--skip-experiments", action="store_true",
                            help="If enabled, will skip experiments with the skip property set to yes")

    args = arg_parser.parse_args()

    config_paths = [Path(path) for path in args.config_paths]
    n_thresholds = int(args.n_thresholds)
    if args.sensitivity_thresholds:
        sensitivity_thresholds = [float(sensitivity_threshold) for sensitivity_threshold in args.sensitivity_thresholds]
    else:
        sensitivity_thresholds = None
    skip_exp = args.skip_experiments

    output_paths: list[Path] = []
    for config_path in config_paths:
        output_path = draw_roc_comparisons(config_path, n_thresholds, sensitivity_thresholds, skip_exp)
        output_paths.append(output_path)

    if len(output_paths) > 1:
        images = [cv2.imread(path.as_posix()) for path in output_paths]
        joint_image = cv2.hconcat(images)
        root = os.path.commonpath(output_paths)
        joint_image_path = Path(root, "joint_roc_comparison.png")
        cv2.imwrite(joint_image_path.as_posix(), joint_image)


if __name__ == "__main__":
    main()
