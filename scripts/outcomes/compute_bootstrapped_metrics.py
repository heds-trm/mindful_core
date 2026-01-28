import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse
from tqdm import tqdm
from typing import Sequence
import warnings

from mindful_core.utils.data_constants import SCAN_ID, SUBSET_ID, LABEL, IN_DISTRIBUTION
from mindful_core.analysis.statistics.roc_compare import delong_roc_variance


class BootstrappedMetric(object):
    def __init__(self, values: Sequence[float] | np.ndarray):
        self.values = np.asarray(values, dtype=np.float32)
        self.mean = np.mean(values)

    def lower_bound(self, alpha: float) -> float:
        return float(np.percentile(self.values, alpha * 100 * 0.5))

    def upper_bound(self, alpha: float) -> float:
        return float(np.percentile(self.values, 100 - alpha * 100 * 0.5))


def compute_eer(ground_truth: np.ndarray,
                probabilities: np.ndarray,
                thresholds_count: int,
                ) -> tuple[float | np.float32, float | np.float32]:
    probabilities_range = probabilities.max() - probabilities.min()
    thresholds = np.arange(thresholds_count) / (thresholds_count - 1)
    thresholds = probabilities.min() + thresholds * probabilities_range
    thresholds = np.expand_dims(thresholds, axis=1)

    probabilities = np.expand_dims(probabilities, axis=0)
    predictions = probabilities >= thresholds
    ground_truth = np.expand_dims(ground_truth, axis=0).astype(bool)

    correct_predictions = predictions == ground_truth

    real_positives_count = np.sum(np.float32(ground_truth))
    real_negatives = ~ground_truth
    real_negatives_count = np.sum(np.float32(real_negatives))

    true_negatives = real_negatives & correct_predictions
    true_negatives_count = np.sum(np.float32(true_negatives), axis=-1)

    true_positives = ground_truth & correct_predictions
    true_positives_count = np.sum(np.float32(true_positives), axis=-1)

    fpr = 1.0 - true_negatives_count / real_negatives_count
    fnr = 1.0 - true_positives_count / real_positives_count

    eer_index = np.argmin(np.abs(fnr - fpr))
    eer_fpr = fpr[eer_index]
    eer_fnr = fnr[eer_index]
    eer = (eer_fpr + eer_fnr) * 0.5
    eer_threshold = thresholds[eer_index]

    # noinspection PyTypeChecker
    return eer, eer_threshold


def compute_bootstrapped_metrics(ground_truth: np.ndarray,
                                 probabilities: np.ndarray,
                                 predictions: np.ndarray = None,
                                 n_iterations: int = 1000,
                                 metrics=("auroc", "eer", "sensitivity", "specificity", "accuracy"),
                                 seed: int = None,
                                 verbose: bool = True
                                 ) -> dict[str, BootstrappedMetric]:
    test_sample_count = len(probabilities)

    if len(ground_truth) != test_sample_count:
        raise RuntimeError("The length of probabilities and ground_truth should match, "
                           "got {} (probabilities) and {} (ground_truth)".
                           format(test_sample_count, len(ground_truth)))

    if predictions is not None:
        if len(predictions) != test_sample_count:
            raise RuntimeError("The length of probabilities and predictions should match, "
                               "got {} (probabilities) and {} (predictions)".
                               format(test_sample_count, len(predictions)))

    random_state = np.random.RandomState(seed)

    ground_truth_bool = ground_truth == 1
    index = np.arange(len(ground_truth))
    positives_index = index[ground_truth_bool]
    negatives_index = index[~ground_truth_bool]

    metrics_values: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in tqdm(range(n_iterations), disable=not verbose):
        positives_sample = random_state.choice(positives_index, size=positives_index.shape[0], replace=True)
        negatives_sample = random_state.choice(negatives_index, size=negatives_index.shape[0], replace=True)
        sample = np.concatenate([positives_sample, negatives_sample])

        ground_truth_sample = ground_truth[sample]
        probabilities_sample = probabilities[sample]

        if "auroc" in metrics:
            auroc_mean, _ = delong_roc_variance(ground_truth_sample, probabilities_sample)
            metrics_values["auroc"].append(auroc_mean)

        eer, eer_threshold = None, None
        if ("eer" in metrics) or (predictions is None):
            eer, eer_threshold = compute_eer(ground_truth_sample, probabilities_sample, thresholds_count=1000)

        if "eer" in metrics:
            metrics_values["eer"].append(float(eer))
        else:
            eer_threshold = None

        if not any([metric in metrics for metric in ("accuracy", "sensitivity", "specificity")]):
            continue

        if predictions is None:
            # if eer_threshold is None:
            #     _, eer_threshold = compute_eer(ground_truth_sample, probabilities_sample, thresholds_count=1000)
            predictions_sample = probabilities_sample > eer_threshold
        else:
            predictions_sample = predictions[sample]

        true_positives = np.sum((ground_truth_sample == 1) & (predictions_sample == 1))  # True Positives
        true_negatives = np.sum((ground_truth_sample == 0) & (predictions_sample == 0))  # True Negatives
        false_positives = np.sum((ground_truth_sample == 0) & (predictions_sample == 1))  # False Positives
        false_negatives = np.sum((ground_truth_sample == 1) & (predictions_sample == 0))  # False Negatives

        if "accuracy" in metrics:
            total_successes = true_positives + true_negatives
            total_samples = true_positives + true_negatives + false_positives + false_negatives
            accuracy = np.float32(total_successes) / np.float32(total_samples)
            metrics_values["accuracy"].append(float(accuracy))

        if "sensitivity" in metrics:
            sensitivity = (np.float32(true_positives) / np.float32(true_positives + false_negatives)
                           if (true_positives + false_negatives) > 0 else 0)
            metrics_values["sensitivity"].append(float(sensitivity))

        if "specificity" in metrics:
            specificity = (np.float32(true_negatives) / np.float32(true_negatives + false_positives)
                           if (true_negatives + false_positives) > 0 else 0)
            metrics_values["specificity"].append(float(specificity))

    results = {metric: BootstrappedMetric(values) for metric, values in metrics_values.items()}
    return results


def compute_eer_threshold_from_available(fold_inferences: pd.DataFrame) -> float:
    # preferred_order = ["test", "validation", "train"]
    preferred_order = ["train", "validation", "test"]
    selected_inferences = None
    for subset_id in preferred_order:
        # noinspection PyTypeChecker
        in_subset: pd.Series = fold_inferences[SUBSET_ID] == subset_id
        if len(in_subset[in_subset]) > 0:
            selected_inferences = fold_inferences[in_subset]
            break

    if selected_inferences is None:
        raise RuntimeError("No valid subset_id found in inferences to compute the EER from. Got: {}".
                           format(fold_inferences.columns))

    probabilities = np.asarray(selected_inferences["Probability"], dtype=float)
    ground_truth = np.asarray(selected_inferences[LABEL], dtype=int)
    _, eer_threshold = compute_eer(ground_truth, probabilities, thresholds_count=1000)
    return eer_threshold


def get_confidence_filter(inferences_path: Path) -> pd.Series | None:
    lightning_folder = inferences_path.parent / "lightning_logs" 
    if not lightning_folder.exists():
        return None
    
    index = int(inferences_path.stem.split("_")[-1])
    version_folder = lightning_folder / "version_{}".format(index)
    confidence_analysis_path = version_folder / "test_positive_threshold_confidence_analysis.csv"

    if not confidence_analysis_path.exists():
        return None
    
    confidence_analysis = pd.read_csv(confidence_analysis_path, index_col=SCAN_ID)
    if IN_DISTRIBUTION not in confidence_analysis.columns:
        warnings.warn("Found a confidence analysis file but could not find the {} column".format(IN_DISTRIBUTION))
        return None
    
    return confidence_analysis[IN_DISTRIBUTION]


def compute_fold_metrics(path: Path,
                         n_iterations: int = 1000,
                         metrics=("auroc", "eer", "sensitivity", "specificity", "accuracy"),
                         seed: int = None,
                         ) -> dict[str, BootstrappedMetric] | None:
    fold_inferences = pd.read_csv(path, index_col=SCAN_ID)

    confidence_filter = get_confidence_filter(path)
    if confidence_filter is not None:
        fold_inferences = fold_inferences[confidence_filter]

    if SUBSET_ID in fold_inferences.columns:
        test_inferences = fold_inferences[fold_inferences[SUBSET_ID] == "test"]
    else:
        test_inferences = fold_inferences

    ground_truth = np.asarray(test_inferences[LABEL], dtype=int)
    probabilities = np.asarray(test_inferences["Probability"], dtype=float)

    if "Prediction (EER)" not in test_inferences.columns:
        return None
        # eer_threshold = compute_eer_threshold_from_available(fold_inferences)
        # predictions = probabilities > eer_threshold
    else:
        predictions = np.asarray(test_inferences["Prediction (EER)"], dtype=int)

    metrics = compute_bootstrapped_metrics(ground_truth, probabilities, predictions, n_iterations, metrics, seed)
    return metrics


def compute_metrics_for_experiment(experiment_path: Path,
                                   n_iterations: int = 1000,
                                   metrics=("auroc", "eer", "sensitivity", "specificity", "accuracy"),
                                   seed: int = None,
                                   ) -> dict[str, BootstrappedMetric] | None:
    folds_metrics = {metric: [] for metric in metrics}
    for inferences_path in experiment_path.glob(pattern="inferences_fold_*.csv"):
        fold_metrics = compute_fold_metrics(inferences_path, n_iterations, metrics, seed)
        if fold_metrics is None:
            return None

        for metric in metrics:
            folds_metrics[metric] += fold_metrics[metric].values.tolist()

    all_metrics = {metric: BootstrappedMetric(values) for metric, values in folds_metrics.items()}
    return all_metrics


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("root", type=str)
    arg_parser.add_argument("--alpha", default=0.05)
    arg_parser.add_argument("--n_iterations", default=1000)
    arg_parser.add_argument("--metrics", nargs="+", default=["auroc", "eer", "sensitivity", "specificity", "accuracy"])
    arg_parser.add_argument("--seed", default=567899)
    arg_parser.add_argument("--save_name", default="bootstrapped_metrics.csv")
    arg_parser.add_argument("--resume", action="store_true", default=False)
    args = arg_parser.parse_args()

    root = Path(args.root)
    alpha = float(args.alpha)
    n_iterations = int(args.n_iterations)
    metrics: list[str] = args.metrics
    seed = int(args.seed)
    save_name: str = args.save_name
    resume: bool = args.resume

    results_by_experiment = {}
    experiment_paths: list[Path] = [inference_file.parent for inference_file in root.rglob("*inferences_fold_00.csv")]
    save_dir = os.path.commonpath(experiment_paths)
    if not save_name.endswith(".csv"):
        save_name = save_name + ".csv"
    save_filepath = Path(save_dir, save_name)

    if resume and save_filepath.exists():
        existing_metrics = pd.read_csv(save_filepath, index_col="ID")
        skip_experiments = existing_metrics.index.tolist()
    else:
        existing_metrics = None
        skip_experiments: list[str] = []

    for experiment_path in tqdm(experiment_paths):
        experiment_name = os.path.relpath(experiment_path, save_dir)
        if experiment_name in skip_experiments:
            continue

        print("Computing metrics for experiment {}".format(experiment_name))

        experiment_metrics = compute_metrics_for_experiment(experiment_path, n_iterations, metrics, seed)
        if experiment_metrics is None:
            warnings.warn("Warning - Missing column `Prediction (EER)` from inferences, "
                          "skipping experiment `{}`".format(experiment_name))
            continue

        experiment_results = {}
        for metric, metric_values in experiment_metrics.items():
            # noinspection PyTypeChecker
            mean = round(metric_values.mean * 100.0, 1)
            lower_bound = round(metric_values.lower_bound(alpha) * 100, 1)
            upper_bound = round(metric_values.upper_bound(alpha) * 100, 1)
            experiment_results[metric] = "{} [{} - {}]".format(mean, lower_bound, upper_bound)

        results_by_experiment[experiment_name] = experiment_results

    results_by_metric = {
        metric: {experiment_name: results_by_experiment[experiment_name][metric]
                 for experiment_name in results_by_experiment}
        for metric in metrics}

    data_frame = pd.DataFrame(results_by_metric)
    data_frame.index.name = "ID"

    if existing_metrics is not None:
        data_frame = pd.concat([existing_metrics, data_frame], axis="index")

    data_frame.to_csv(save_filepath)


if __name__ == "__main__":
    main()
