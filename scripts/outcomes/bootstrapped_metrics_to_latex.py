import argparse
from pathlib import Path
import pandas as pd

from mindful_core.utils.latex import pandas_to_latex


METRIC_NAMES = {
    "auroc": "AUROC",
    "eer": "EER",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "accuracy": "Accuracy"
}


def format_metric(metric: str) -> str:
    metric = metric.replace("[", "\medspace [")
    metric = "${}$".format(metric)
    return metric


def add_experiments_info(data_frame: pd.DataFrame,
                         experiments_info_path: str | Path | pd.DataFrame | None = None
                         ) -> tuple[pd.DataFrame, int]:
    if experiments_info_path is None:
        return data_frame, 0
    
    if isinstance(experiments_info_path, pd.DataFrame):
        experiments_info = experiments_info_path
    else:
        experiments_info = pd.read_csv(experiments_info_path, index_col=data_frame.index.name)

    data_frame = pd.concat([experiments_info, data_frame], axis="columns")
    return data_frame, len(experiments_info.columns)


def bootstrapped_metrics_to_latex(path_or_data_frame: str | Path | pd.DataFrame,
                                  experiments_info_path: str | Path | pd.DataFrame | None = None,
                                  ) -> str:
    if not isinstance(path_or_data_frame, pd.DataFrame):
        data = pd.read_csv(path_or_data_frame, index_col=0)
    else:
        data = path_or_data_frame.copy()

    data = data.applymap(format_metric)
    data, added_columns = add_experiments_info(data, experiments_info_path)
    data.index = data.index.map(lambda idx: str(idx).replace("_", "-"))
    data = data.rename(METRIC_NAMES, axis="columns")

    if experiments_info_path is None:
        ignore_index = False
        left_count = 1
    else:
        ignore_index = True
        left_count = added_columns
    latex_table = pandas_to_latex(data, ignore_index=ignore_index, left_count=left_count)
    
    return latex_table

    
def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("paths", nargs="+")
    arg_parser.add_argument("--silent", action="store_true")
    arg_parser.add_argument("--default_name", default="bootstrapped_metrics.csv")
    arg_parser.add_argument("--use_experiments_info", action="store_true")
    arg_parser.add_argument("--experiments_info_name", default="experiments_info.csv")
    args = arg_parser.parse_args()

    verbose = not args.silent
    default_name: str = args.default_name
    use_experiments_info: bool = args.use_experiments_info
    experiments_info_name: str = args.experiments_info_name
    paths: list[Path] = [Path(path) for path in args.paths]
    updated_paths: list[Path] = []
    missing_paths: list[Path] = []
    
    # region Find files
    for path in paths:
        if not path.exists():
            missing_paths.append(path)

        elif path.is_dir():
            sub_paths = [sub_path for sub_path in path.rglob(default_name) if sub_path.is_file()]
            updated_paths += sub_paths

        else:
            updated_paths.append(path)
    # endregion

    if len(missing_paths) > 0:
        raise FileNotFoundError("The following path(s) were missing: {}".format(missing_paths))
    
    for path in updated_paths:
        target_path = path.with_suffix(".tex")

        if verbose:
            print("Converting `{}` to LaTeX table in `{}`".format(path, target_path))

        experiments_info_path = path.parent / experiments_info_name
        if (not experiments_info_path.exists()) or not use_experiments_info:
            experiments_info_path = None

        latex = bootstrapped_metrics_to_latex(path, experiments_info_path)
        with open(target_path, "w") as target_file:
            target_file.write(latex)


if __name__ == "__main__":
    main()
