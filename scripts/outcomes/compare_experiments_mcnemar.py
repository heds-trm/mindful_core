import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2
from pathlib import Path
import argparse
import os
from typing import Union


class McNemarTest(object):
    def __init__(self, statistic: float, pvalue: float, best: str):
        self.statistic = statistic
        self.pvalue = pvalue
        self.best = best

    def format_test(self, a_id: str, b_id: str) -> str:
        pvalue = "={}".format(round(self.pvalue, 5)) if self.pvalue >= 1e-5 else "<1e-5"
        if self.best == "a":
            best_id = a_id
        elif self.best == "b":
            best_id = b_id
        else:
            best_id = "Mixed"
        return "{} (pvalue{})".format(best_id, pvalue)

    @staticmethod
    def from_data_frames(experiment_a: pd.DataFrame,
                         experiment_b: pd.DataFrame,
                         criterion: str) -> Union["McNemarTest", None]:
        if (criterion not in experiment_a) or (criterion not in experiment_b):
            return None

        index = experiment_a.index
        values_a = experiment_a[criterion][index]
        values_b = experiment_b[criterion][index]

        both_right = values_a & values_b
        a_right = values_a & ~values_b
        b_right = ~values_a & values_b
        both_wrong = ~values_a & ~values_b

        comparison_table = [[both_wrong.sum(), a_right.sum()],
                            [b_right.sum(), both_right.sum()]]

        best = "a" if a_right.sum() > b_right.sum() else "b"
        result = mcnemar(comparison_table)

        return McNemarTest(result.statistic, result.pvalue, best)


class McNemarNaiveCrossValidationTest(object):
    def __init__(self, individual_tests: list[McNemarTest]):
        self.individual_tests = individual_tests

    def aggregate(self) -> McNemarTest:
        statistic = sum([test.statistic for test in self.individual_tests])
        pvalue = chi2.sf(statistic, df=len(self.individual_tests))

        if all([test.best == "a" for test in self.individual_tests]):
            best = "a"
        elif all([test.best == "b" for test in self.individual_tests]):
            best = "b"
        else:
            best = "mixed"

        return McNemarTest(statistic, pvalue, best)


def load_experiments_inferences(experiments_paths: list[Path]) -> dict[str, list[pd.DataFrame]]:
    # common_path = os.path.commonpath(experiments_paths)
    experiments_inferences: dict[str, list[pd.DataFrame]] = {}
    for experiment_path in experiments_paths:
        # experiment_id = os.path.relpath(experiment_path, common_path)
        experiment_id = experiment_path.name

        experiment_inferences = {}
        for fold_inferences_path in experiment_path.glob("*inferences_fold_*.csv"):
            fold_inferences = pd.read_csv(fold_inferences_path, index_col="ScanID")
            if "SubsetID" in fold_inferences:
                fold_inferences = fold_inferences[fold_inferences["SubsetID"] == "test"]
            fold_id = int(fold_inferences_path.stem.split("inferences_fold_")[-1])
            experiment_inferences[fold_id] = fold_inferences

        sorted_experiment_inferences = [experiment_inferences[key] for key in
                                        sorted(list(experiment_inferences.keys()))]
        experiments_inferences[experiment_id] = sorted_experiment_inferences

    return experiments_inferences


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("experiment_paths", nargs="+")
    arg_parser.add_argument("--output_name", default=None)
    arg_parser.add_argument("--output_dir", default=None)
    args = arg_parser.parse_args()

    experiments_paths = [Path(path) for path in args.experiment_paths]
    output_name = args.output_name
    output_dir = Path(os.path.commonpath(experiments_paths) if args.output_dir is None else args.output_dir)

    experiments_inferences = load_experiments_inferences(experiments_paths)

    potential_metrics = {
        "EER": "Correctness (EER)",
        "Sensitivity": "Correctness (Sens.)",
        "Accuracy": "Correctness (Acc.)",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Saving McNemar outcomes to the following folder: {}".format(output_dir))

    for metric_name, metric_column in potential_metrics.items():
        results = {xp_b_id: {xp_a_id: None for xp_a_id in experiments_inferences}
                   for xp_b_id in experiments_inferences}
        metric_not_empty = False
        for i, (xp_a_id, xp_a) in enumerate(experiments_inferences.items()):
            for j, (xp_b_id, xp_b) in enumerate(experiments_inferences.items()):
                if i >= j:
                    continue

                mcnemar_results = [McNemarTest.from_data_frames(xp_a_fold, xp_b_fold, criterion=metric_column)
                                   for xp_a_fold, xp_b_fold in zip(xp_a, xp_b)]
                result_is_none = [result is None for result in mcnemar_results]
                if any(result_is_none):
                    if not all(result_is_none):
                        raise RuntimeError("Either all or none outcomes should be none")
                    continue

                metric_not_empty = True

                mcmenar_cross_val_result = McNemarNaiveCrossValidationTest(mcnemar_results)
                # noinspection PyTypeChecker
                results[xp_b_id][xp_a_id] = mcmenar_cross_val_result.aggregate().format_test(xp_a_id, xp_b_id)

        if metric_not_empty:
            filename = "mcnemar_{}.csv".format(metric_name)
            if output_name is not None:
                filename = "{}_{}".format(output_name, filename)
            pd.DataFrame(results).to_csv(output_dir / filename)


if __name__ == "__main__":
    main()
