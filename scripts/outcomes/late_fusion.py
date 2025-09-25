import torch
import pandas as pd
from pathlib import Path
import argparse

from mindful_core.analysis.statistics.classification_summary import (
    ClassificationSummary,

    AUROC,
    AccuracyAtEER,

    EER,
    SpecificityAtEER,
    SensitivityAtEER,

    make_metrics_for_sensitivity_threshold,
    SensitivityThreshold,
    AccuracyAtSensitivity,
    SensitivityAtSensitivity,
    SpecificityAtSensitivity,
)
from mindful_core.scripts.outcomes.gather_test_results import summarize_experiments
from mindful_core.utils.misc import save_csv


def make_classification_summary(sensitivity_thresholds: list[float]
                                ) -> ClassificationSummary:
    metrics = [
        AUROC,
        EER,
        AccuracyAtEER,

        SensitivityAtEER,
        SpecificityAtEER,
    ]

    for sensitivity_threshold in sensitivity_thresholds:
        sensitivity_metrics = make_metrics_for_sensitivity_threshold(base_metrics=[
            SensitivityThreshold,
            AccuracyAtSensitivity,
            SensitivityAtSensitivity,
            SpecificityAtSensitivity
        ],
            relative_threshold=sensitivity_threshold)

        sensitivity_threshold_metrics = sensitivity_metrics.pop(0)
        metrics += sensitivity_threshold_metrics

    classification_summary = ClassificationSummary(metrics)
    return classification_summary


def load_test_folds(folds_path: str | Path) -> dict[str, list]:
    folds_path = Path(folds_path)

    test_folds = {}
    for fold_path in folds_path.glob(pattern="*fold_*.csv"):
        fold = pd.read_csv(fold_path, index_col="ScanID")
        if "SubsetID" in fold:
            fold = fold[fold["SubsetID"] == "test"]
        fold_id = fold_path.stem[-2:]
        test_folds[fold_id] = fold.index.tolist()
    return test_folds


def infer_test_folds(experiment_path: Path | list[Path]) -> dict[str, list]:
    if isinstance(experiment_path, (list, tuple)):
        for path in experiment_path:
            path = Path(path)
            if path.is_dir():
                return infer_test_folds(path)
        raise RuntimeError("Could not infer folds from given paths.")

    return load_test_folds(experiment_path)


def get_test_folds(inferences_paths: list[Path], folds_path: str | Path = None) -> dict[str, list] | None:
    if requires_folds(inferences_paths):
        if folds_available(inferences_paths, folds_path):
            if folds_path is not None:
                test_folds = load_test_folds(folds_path)
            else:
                test_folds = infer_test_folds(inferences_paths)
        else:
            raise ValueError("When providing only unfolded inference files, you must provide the path to folds "
                             "to allow for separating probabilities by fold.")
    else:
        test_folds = None

    return test_folds


def requires_folds(inferences_paths: list[Path]) -> bool:
    return any([inferences_path.is_file() for inferences_path in inferences_paths])


def folds_available(inferences_paths: list[Path], folds_path: str | Path = None) -> bool:
    return ((folds_path is not None) or
            any([inferences_path.is_dir() for inferences_path in inferences_paths]))


def get_folded_experiment_probabilities(experiment_path: Path) -> dict[str, pd.Series]:
    probabilities = {}

    for fold_path in experiment_path.glob(pattern="*_fold_*.csv"):
        fold = pd.read_csv(fold_path, index_col="ScanID")
        test_inferences = fold[fold["SubsetID"] == "test"]
        fold_probabilities = test_inferences["Probability"]

        fold_id = fold_path.stem[-2:]
        if fold_id not in probabilities:
            probabilities[fold_id] = {}

        probabilities[fold_id] = fold_probabilities

    return probabilities


def get_unfolded_experiment_probabilities(inference_path: Path,
                                          test_folds: dict[str, list]
                                          ) -> dict[str, pd.Series]:
    inferences = pd.read_csv(inference_path, index_col="ScanID")
    probabilities = {fold_id: inferences.loc[fold_indices]["Probability"]
                     for fold_id, fold_indices in test_folds.items()}

    return probabilities


def get_probabilities(inferences_paths: list[Path], folds_path: str | Path = None) -> dict[str, pd.DataFrame]:
    inferences_paths = [Path(inference_path) for inference_path in inferences_paths]
    probabilities: dict[str, dict[str, pd.Series]] = {}

    test_folds = get_test_folds(inferences_paths, folds_path)

    for inference_path in inferences_paths:
        if inference_path.is_dir():
            experiment_probabilities = get_folded_experiment_probabilities(inference_path)
        else:
            experiment_probabilities = get_unfolded_experiment_probabilities(inference_path, test_folds)

        for fold_id, fold_probabilities in experiment_probabilities.items():
            if fold_id not in probabilities:
                probabilities[fold_id] = {}
            probabilities[fold_id][inference_path.as_posix()] = fold_probabilities

    result = {fold_id: pd.DataFrame(fold_probabilities)
              for fold_id, fold_probabilities in probabilities.items()}
    return result


def get_labels(inference_path: Path) -> dict[str, pd.Series]:
    inference_path = Path(inference_path)
    labels: dict[str, pd.Series] = {}
    for fold_path in inference_path.glob(pattern="*_fold_*.csv"):
        fold = pd.read_csv(fold_path, index_col="ScanID")
        fold_id = fold_path.stem[-2:]
        test_fold = fold[fold["SubsetID"] == "test"]
        labels[fold_id] = test_fold["Label"]

    return labels


def late_fusion(inferences: list[str | Path],
                sensitivity_thresholds: list[float],
                output_path: str | Path = None):
    probabilities = get_probabilities(inferences)
    labels = get_labels(inferences[0])
    classification_summary = make_classification_summary(sensitivity_thresholds)

    results = {}
    for fold_id in probabilities:
        fold_probabilities = probabilities[fold_id]
        unweighted_joint = fold_probabilities.mean(axis=1)
        fold_labels = labels[fold_id][fold_probabilities.index]

        unweighted_joint = torch.as_tensor(unweighted_joint.array)
        fold_labels = torch.as_tensor(fold_labels.array, dtype=torch.int32)

        fold_results = classification_summary(unweighted_joint, fold_labels, probabilities=unweighted_joint)
        fold_results = {metric_name: float(metric_value) for metric_name, metric_value in fold_results.items()}

        results[fold_id] = fold_results

    gathered_results = pd.DataFrame.from_dict(results, orient="index")
    gathered_results.index.name = "Test ID"

    fold_indices = [[fold_id for fold_id in probabilities.keys()]]
    # noinspection PyTypeChecker
    summary = summarize_experiments(gathered_results, indices=fold_indices)

    save_csv(output_path, summary, fallback_name="late_fusion", save_index=False)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("inferences", nargs="+")
    arg_parser.add_argument("--sensitivity_thresholds", nargs="+", default=None)
    arg_parser.add_argument("--output", required=True)
    arg_parser.add_argument("--folds", default=None)
    args = arg_parser.parse_args()

    inferences: list[str] = args.inferences
    sensitivity_thresholds = ([] if args.sensitivity_thresholds is None
                              else [float(x) for x in args.sensitivity_thresholds])
    output: str = args.output
    # folds: str | None = args.folds

    late_fusion(inferences, sensitivity_thresholds, output)


if __name__ == "__main__":
    main()
