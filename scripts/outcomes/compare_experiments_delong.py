import pandas as pd
from pathlib import Path
import argparse
from typing import Any

from mindful_core.analysis.statistics.roc_compare import compute_kfolds_delong_roc_test
from mindful_core.utils.misc import load_json
from mindful_core.scripts.outcomes.draw_roc_comparisons import get_test_partitions, extract_ground_truth_and_probabilities


def load_test_inferences_from_file(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        # noinspection PyTypeChecker
        data_frame = pd.read_excel(path, index_col="ScanID")
    else:
        data_frame = pd.read_csv(path, index_col="ScanID")
    return data_frame


def load_test_inferences_from_config(config: str | Path | dict[str, Any]) -> dict[str, pd.DataFrame]:
    if not isinstance(config, dict):
        config = load_json(config)
    return {exp_id: load_test_inferences_from_file(path)
            for exp_id, path in config.items()
            if not exp_id.startswith("-")}


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_path", type=str)
    args = arg_parser.parse_args()

    config_path = Path(args.config_path)
    config = load_json(config_path)
    folds_ids = get_test_partitions(config["folds"])
    test_inferences = load_test_inferences_from_config(config["experiments"])

    ground_truth, test_probabilities = extract_ground_truth_and_probabilities(test_inferences, folds_ids)
    delong_summary_data: dict[str, dict[str, bool]] = {}
    delong_pairs_data: dict[str, list] = {}
    for i, (exp_id_i, exp_probabilities_i) in enumerate(test_probabilities.items()):
        for j, (exp_id_j, exp_probabilities_j) in enumerate(test_probabilities.items()):
            if i >= j:
                continue

            (reject, p_values), (reject_corrected, p_values_corrected) = \
                compute_kfolds_delong_roc_test(ground_truth, exp_probabilities_i, exp_probabilities_j)
            if exp_id_i not in delong_summary_data:
                delong_summary_data[exp_id_i] = {}
            # noinspection PyTypeChecker
            delong_summary_data[exp_id_i][exp_id_j] = reject_corrected.all()

            pair_id = "{} // {}".format(exp_id_i, exp_id_j)
            pair_data = (
                    list(p_values) +
                    list(reject) +
                    [reject.astype(int).sum()] +
                    list(p_values_corrected) +
                    list(reject_corrected) +
                    [reject_corrected.astype(int).sum()]
            )
            delong_pairs_data[pair_id] = pair_data

    delong_summary = pd.DataFrame.from_dict(delong_summary_data, orient="index")
    summary_output_filepath = config_path.parent / "delong_comparison_summary.csv"
    delong_summary.to_csv(summary_output_filepath)

    pairs_columns = (
            ["p_value_{}".format(i) for i in range(len(folds_ids))] +
            ["reject_{}".format(i) for i in range(len(folds_ids))] +
            ["reject count"] +
            ["p_value_{} (corrected)".format(i) for i in range(len(folds_ids))] +
            ["reject_{} (corrected)".format(i) for i in range(len(folds_ids))] +
            ["reject count (corrected)"]
    )
    delong_pairs = pd.DataFrame.from_dict(delong_pairs_data, orient="index", columns=pairs_columns)
    pairs_output_filepath = config_path.parent / "delong_comparison_pairs.csv"
    delong_pairs.to_csv(pairs_output_filepath)


if __name__ == "__main__":
    main()
