import argparse
from pathlib import Path
import pandas as pd

TABLE_TEMPLATE = """
\\begin{{table}}[H]
    \centering
    \\begin{{tabular}}{{{alignement}}}
        \\toprule
        {header} \\\\
        \\midrule
        {rows} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{Placeholder caption.}}
    \\label{{tab:placeholder_label}}
\\end{{table}}
"""

def format_metric(metric: str) -> str:
    metric = metric.replace("[", "\medspace [")
    metric = "${}$".format(metric)
    return metric


def bootstrapped_metrics_to_latex(path_or_data_frame: str | Path | pd.DataFrame) -> str:
    if not isinstance(path_or_data_frame, pd.DataFrame):
        data = pd.read_csv(path_or_data_frame, index_col="ID")
    else:
        data = path_or_data_frame

    alignement = "|l|" + "c|" * len(data.columns)
    latex_header = " & " + " & ".join(data.columns)
    latex_rows: list[str] = []

    for row_id, row_data in data.iterrows():
        row_id = row_id.replace("_", "-")
        row_data = [format_metric(row_data[column]) for column in data.columns]
        latex_row = " & ".join([row_id, *row_data])
        latex_rows.append(latex_row)

    latex_rows_joint = "\\\\\n\t\t".join(latex_rows)
    latex_table = TABLE_TEMPLATE.format(alignement=alignement, header=latex_header, rows=latex_rows_joint)

    return latex_table

    
def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("paths", nargs="+")
    arg_parser.add_argument("--silent", action="store_true")
    arg_parser.add_argument("--default_name", default="bootstrapped_metrics.csv")
    args = arg_parser.parse_args()

    verbose = not args.silent
    default_name: str = args.default_name
    paths: list[Path] = [Path(path) for path in args.paths]
    updated_paths: list[Path] = []
    missing_paths: list[Path] = []
    
    for path in paths:
        if not path.exists():
            missing_paths.append(path)

        elif path.is_dir():
            sub_paths = [sub_path for sub_path in path.rglob(default_name) if sub_path.is_file()]
            updated_paths += sub_paths

        else:
            updated_paths.append(path)

    if len(missing_paths) > 0:
        raise FileNotFoundError("The following path(s) were missing: {}".format(missing_paths))
    
    for path in updated_paths:
        target_path = path.with_suffix(".tex")

        if verbose:
            print("Converting `{}` to LaTeX table in `{}`".format(path, target_path))

        latex = bootstrapped_metrics_to_latex(path)
        with open(target_path, "w") as target_file:
            target_file.write(latex)


if __name__ == "__main__":
    main()
