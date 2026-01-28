import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from mindful_core.utils.misc import is_defined


METRIC_NAMES = {
    "auroc": "AUROC",
    "eer": "EER",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "accuracy": "Accuracy"
}
EOL = " \\\\"
TOP_RULE = "\\toprule"
MID_RULE = "\\midrule"
BOT_RULE = "\\bottomrule"


def get_row_start(row_id: str, row_starts: dict[str, dict[str, str]] | None) -> list[str]:
    if row_starts is None:
        return [row_id]
    
    row_start = []
    for _, starts in row_starts.items():
        if row_id in starts:
            row_start.append(starts[row_id])

    if len(row_start) == 0:
        return [row_id]
    
    return row_start

def process_metric_value(value):
    if is_defined(value):
        return round(value * 100.0, 1)
    else:
        return value


def get_summary_best_values(summary: pd.DataFrame) -> dict[str, float]:
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


def format_experiments_values(metric_mean, metric_std, bold_font: bool) -> str:
    if not (is_defined(metric_mean) and is_defined(metric_std)):
        return ""
    else:
        metric_mean = process_metric_value(metric_mean)
        metric_std = process_metric_value(metric_std)

        if bold_font:
            metric_mean = "\\bm{{{}}}".format(metric_mean)

        return "${}\\pm{}$".format(metric_mean, metric_std)


def experiments_summary_to_latex(summary: pd.DataFrame, row_starts: dict[str, dict[str, str]] | None = None) -> list[str]:
    metrics = [column[:-5] for column in summary.columns if column.endswith("_mean")]
    metrics_fields = [(metric + "_mean", metric + "_std") for metric in metrics]

    latex_rows = []
    best_values = get_summary_best_values(summary)

    for name, summary_row in summary.iterrows():
        latex_row = get_row_start(name, row_starts)

        for metric_name, (metric_mean_field, metric_std_field) in zip(metrics, metrics_fields):
            metric_mean = summary_row[metric_mean_field]
            metric_std = summary_row[metric_std_field]
            use_bold_font = is_best_value(metric_mean, best_values[metric_name])

            metric_text = format_experiments_values(metric_mean, metric_std, use_bold_font)
            latex_row.append(metric_text)

        latex_rows.append(" & ".join(latex_row) + EOL)

    return latex_rows


def get_bootstrapped_best_values(summary: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    def parse_metric_mean(text: str) -> float:
        mean = text.split(" ")[0]
        return float(mean)
    
    summary_mean: pd.DataFrame = summary.applymap(parse_metric_mean)
    summary_renamed = summary_mean.copy()
    summary_renamed.columns = summary_mean.columns.map(lambda name: name + "_mean")
    best_values = get_summary_best_values(summary_renamed)

    return best_values, summary_mean


def bootstrapped_summary_to_latex(summary: pd.DataFrame, row_starts: dict[str, dict[str, str]] | None = None) -> list[str]:
    # region Header
    latex_rows = [TOP_RULE]

    metrics = [METRIC_NAMES.get(name, name) for name in summary.columns]
    header = " & ".join([*row_starts.keys(), *metrics])
    latex_rows.append(header + EOL)

    latex_rows.append(MID_RULE)
    # endregion

    best_values, summary_mean = get_bootstrapped_best_values(summary)
    for row_id, summary_row in summary.iterrows():
        latex_row = get_row_start(row_id, row_starts)

        for metric_name in summary.columns:
            metric_value = summary_row[metric_name]
            metric_mean = summary_mean[metric_name][row_id]
            use_bold_font = is_best_value(metric_mean, best_values[metric_name])
            if use_bold_font:
                metric_value = "\\bm{{{}}}".format(metric_value)
            metric_value = "${}$".format(metric_value)
            latex_row.append(metric_value)

        latex_rows.append(" & ".join(latex_row) + EOL)

    latex_rows.append(BOT_RULE)

    return latex_rows


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("summary", type=str)
    arg_parser.add_argument("--row_starts", type=str, default=None)
    args = arg_parser.parse_args()

    summary_path = Path(args.summary)
    summary = pd.read_csv(summary_path, index_col=0)

    if args.row_starts is not None:
        row_starts_path = Path(args.row_starts)
        row_starts = pd.read_csv(row_starts_path, index_col=0)
        row_starts = row_starts.to_dict()
    else:
        row_starts = None

    is_experiments_summary = any(["_mean" in column for column in summary.columns])

    if is_experiments_summary:
        rows = experiments_summary_to_latex(summary, row_starts)
    else:
        rows = bootstrapped_summary_to_latex(summary, row_starts)

    with open(summary_path.parent / "{}_latex.txt".format(summary_path.stem), "w") as output_file:
        output_file.write("\n".join(rows))


if __name__ == "__main__":
    main()
