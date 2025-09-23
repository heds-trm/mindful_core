import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from utils.misc import is_defined


def process_metric_value(value):
    if is_defined(value):
        return round(value * 100.0, 1)
    else:
        return value


def metric_values_to_latex(metric_mean, metric_std, bold_font: bool) -> str:
    if not (is_defined(metric_mean) and is_defined(metric_std)):
        return ""
    else:
        metric_mean = process_metric_value(metric_mean)
        metric_std = process_metric_value(metric_std)

        if bold_font:
            metric_mean = "\\bm{{{}}}".format(metric_mean)

        return "${}\\pm{}$".format(metric_mean, metric_std)


def get_summary_best_values(summary: pd.DataFrame) -> dict[str, int]:
    maximums = summary.max(numeric_only=True, skipna=True)
    minimums = summary.min(numeric_only=True, skipna=True)

    best_values = {}
    column: str
    for column in summary:
        if not column.endswith("_mean"):
            continue
        metric_name = column[:-5]
        if metric_name in ["eer"]:
            best_value = minimums[column]
        else:
            best_value = maximums[column]
        best_values[metric_name] = best_value
    return best_values


def is_best_value(value: float, reference: float) -> bool:
    return np.abs(value - reference) < 0.001


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--summary", type=str, required=True)
    args = arg_parser.parse_args()

    summary_path = Path(args.summary)
    summary = pd.read_csv(summary_path)

    metrics = [column[:-5] for column in summary.columns if column.endswith("_mean")]
    metrics_fields = [(metric + "_mean", metric + "_std") for metric in metrics]

    rows = []
    best_values = get_summary_best_values(summary)
    for i in range(len(summary)):
        row = [summary.loc[i]["ID"]]
        for metric_name, (metric_mean_field, metric_std_field) in zip(metrics, metrics_fields):
            metric_mean = summary.loc[i][metric_mean_field]
            metric_std = summary.loc[i][metric_std_field]
            use_bold_font = is_best_value(metric_mean, best_values[metric_name])

            metric_text = metric_values_to_latex(metric_mean, metric_std, use_bold_font)
            row.append(metric_text)
        rows.append(" & ".join(row) + " \\\\")

    with open(summary_path.parent / "experiments_summary_latex.txt", "w") as output_file:
        output_file.write("\n".join(rows))


if __name__ == "__main__":
    main()
