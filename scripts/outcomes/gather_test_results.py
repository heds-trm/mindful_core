import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
import os
from typing import Optional, Iterator


from utils.metrics import format_metric


def find_all_test_results_files(path: str | Path) -> list[str]:
    test_results_files = []
    for folder, _, files in os.walk(path):
        if "test_results.csv" in files:
            test_results_files.append(os.path.join(folder, "test_results.csv"))
    return test_results_files


def _load_test_results(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as file:
        header = next(file).split(",")
        transpose = (len(header) > 1) and header[1].isnumeric()

    if transpose:
        data = pd.read_csv(filepath, index_col=0, header=None)
        data = data.transpose()
    else:
        data = pd.read_csv(filepath)

    data.columns = data.columns.str.replace("test_", "")
    data.columns = data.columns.str.replace("_epoch", "")
    data.columns = data.columns.str.replace("auc_roc", "auroc")
    return data.squeeze()


def load_tests_results(test_results_files: list[str]):
    test_results = [_load_test_results(filepath) for filepath in test_results_files]
    test_results = pd.DataFrame(test_results).reset_index(drop=True).fillna("")

    for column_id in test_results.columns:
        if "auroc" in column_id:
            column_values = test_results[column_id]
            float_values = [isinstance(column_value, float) for column_value in column_values]
            test_results.loc[float_values, column_id] = column_values[float_values].clip(0.5, 1.0)

    return test_results


def get_test_ids(test_results_files: list[str], save_absolute_path: bool) -> pd.DataFrame:
    if save_absolute_path:
        test_ids = test_results_files
    else:
        root = os.path.commonpath(test_results_files)
        test_ids = [os.path.relpath(filepath, root) for filepath in test_results_files]

    return pd.DataFrame(data=test_ids, columns=["Test ID"])


def list_experiments(test_results_files: list[str], min_folds: int = -1) -> dict[str, list[int]]:
    experiments: dict[str, list[int]] = {}
    for test_index, test_result_file in enumerate(test_results_files):
        experiment_path = get_experiment_path(test_result_file)
        if experiment_path not in experiments:
            experiments[experiment_path] = []
        experiments[experiment_path].append(test_index)
    
    experiments = {experiment_path: indices 
                   for experiment_path, indices 
                   in experiments.items()
                   if len(indices) >= min_folds}
    return experiments


def summarize_experiments(test_results: pd.DataFrame,
                          indices: Iterator[list]
                          ) -> pd.DataFrame:
    experiments_summaries = []
    for test_indices in indices:
        experiment_results = test_results.loc[test_indices]

        columns_with_data = experiment_results.any()
        kept_columns = [column_name for column_name, keep in columns_with_data.items() if keep]
        experiment_results = experiment_results[kept_columns]

        experiment_mean = experiment_results.mean().add_suffix("_mean")
        experiment_std = experiment_results.std(ddof=0).add_suffix("_std")

        experiment_summary = pd.concat([experiment_mean, experiment_std])
        experiments_summaries.append(experiment_summary)

    experiments_summary = pd.DataFrame(experiments_summaries)
    return experiments_summary


def add_experiments_headers(experiments_summary: pd.DataFrame,
                            experiments_paths: Iterator[str],
                            save_absolute_path) -> pd.DataFrame:
    if save_absolute_path:
        ids = list(experiments_paths)
    else:
        # noinspection PyTypeChecker
        root = os.path.commonpath(experiments_paths)
        # noinspection PyTypeChecker
        ids = [os.path.relpath(experiment_path, root) for experiment_path in experiments_paths]
    timestamps = [get_experiment_timestamp(experiment_path) for experiment_path in experiments_paths]
    dates = [datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') for timestamp in timestamps]

    experiments_headers = pd.DataFrame(data={"ID": ids, "Time": dates})
    experiments_summary = pd.concat([experiments_headers, experiments_summary], axis=1)
    experiments_summary = experiments_summary.reindex(np.argsort(timestamps))

    return experiments_summary


def sort_experiments_summary_columns(experiments_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    other_columns = []
    mean_tag = "mean"
    std_tag = "std"
    tag_length = len(mean_tag)
    for column in experiments_summary.columns:
        column: str
        if column.endswith(mean_tag):
            metrics.append(column[:-tag_length])
        elif not column.endswith(std_tag):
            other_columns.append(column)

    columns = sum([[column + suffix for column in metrics] for suffix in [mean_tag, std_tag]], other_columns)
    return experiments_summary.reindex(columns=columns)


def format_experiments_summary(experiments_summary: pd.DataFrame) -> pd.DataFrame:
    formatted_data = {}

    for column in experiments_summary.columns:
        std_name = column.replace("_mean", "_std")
        if column.endswith("_mean") and (std_name in experiments_summary.columns):
            metric_name = column[:-len("_mean")]
            metric_values = experiments_summary[[column, std_name]].agg(format_metric, axis=1)
            formatted_data[metric_name] = metric_values
        else:
            mean_name = column.replace("_std", "_mean")
            if not (column.endswith("_std") and (mean_name in experiments_summary.columns)):
                formatted_data[column] = experiments_summary[column]

    return pd.DataFrame(formatted_data, index=experiments_summary.index)


def get_experiment_path(test_result_file: str) -> str:
    # test_result_file : experiment_path/lightning_logs/version_{}/test_results.csv
    return os.path.dirname(os.path.dirname(os.path.dirname(test_result_file)))


def get_experiment_timestamp(experiment_path: str) -> float:
    config_filepath = Path(experiment_path) / "lightning_logs" / "version_0" / "experiment_config.json"
    return config_filepath.stat().st_mtime


def build_experiments_summary(experiments: dict[str, list[int]], test_results: pd.DataFrame, save_absolute_path: bool):
    # noinspection PyTypeChecker
    experiments_summary = summarize_experiments(test_results, experiments.values()).reset_index(drop=True)
    # noinspection PyTypeChecker
    experiments_summary = add_experiments_headers(experiments_summary, experiments, save_absolute_path)
    experiments_summary = sort_experiments_summary_columns(experiments_summary)
    return experiments_summary


def build_mann_whitney_summary(experiments: dict[str, list[int]],
                               test_results: pd.DataFrame
                               ) -> pd.DataFrame:
    shared_path = os.path.commonpath(list(experiments.keys()))
    auroc_results = test_results["auroc"]
    mann_whitney_entries = []
    for i, (experiment_a, indices_a) in enumerate(experiments.items()):
        auroc_a = auroc_results.loc[indices_a].to_numpy()
        for j, (experiment_b, indices_b) in enumerate(experiments.items()):
            if (j <= i) or (len(indices_b) < len(indices_a)):
                continue
            auroc_b = auroc_results.loc[indices_b].to_numpy()
            mann_whitney_result = mannwhitneyu(auroc_a, auroc_b)
            t_test_ind_result = ttest_ind(auroc_a, auroc_b, equal_var=False)
            mann_whitney_entry = {
                "Experiment A": os.path.relpath(experiment_a, shared_path),
                "Experiment B": os.path.relpath(experiment_b, shared_path),
                "MW Statistic": mann_whitney_result.statistic,
                "MW p-value": mann_whitney_result.pvalue,
                "MW significant": mann_whitney_result.pvalue <= 0.05,

                "TTest Statistic": t_test_ind_result.statistic,
                "TTest p-value": t_test_ind_result.pvalue,
                "TTest significant": t_test_ind_result.pvalue <= 0.05
            }
            mann_whitney_entries.append(mann_whitney_entry)
    mann_whitney_summary = pd.DataFrame(mann_whitney_entries)
    return mann_whitney_summary


def add_datasets_column(data_frame: pd.DataFrame,
                        datasets: list[str],
                        index_col: str = None,
                        new_col: str = "Dataset",
                        new_col_loc: int = 1
                        ) -> pd.DataFrame:
    def get_dataset_from_id(row_id: str) -> str:
        for dataset in datasets:
            if dataset in row_id:
                return dataset
        return ""

    if index_col is None:
        data_frame.index.map(get_dataset_from_id)
    else:
        dataset_by_id = data_frame[index_col].apply(get_dataset_from_id)
        data_frame = data_frame.copy()
        data_frame.insert(loc=new_col_loc, column=new_col, value=dataset_by_id)

    return data_frame


def add_experiment_id_from_index(experiments: dict[str, list[int]],
                                 data_frame: pd.DataFrame,
                                 ):
    def get_experiment_id_from_index(row_index: int) -> str:
        for experiment_path, indices in experiments.items():
            if row_index in indices:
                return experiment_path
        raise ValueError

    ids = data_frame.index.map(get_experiment_id_from_index)

    data_frame = data_frame.copy()
    data_frame.insert(loc=0, column="ID", value=ids)

    return data_frame


def export_summaries(experiments: dict[str, list[int]],
                     test_results: pd.DataFrame,
                     save_absolute_path: bool,
                     save_folder: str = None,
                     datasets: list[str] = None,
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiments_summary_path = os.path.join(save_folder, "experiments_summary.csv")
    experiments_summary = build_experiments_summary(experiments, test_results, save_absolute_path)
    experiments_summary.to_csv(experiments_summary_path, index=False)

    formatted_summary_path = os.path.join(save_folder, "formatted_summary.csv")
    formatted_summary = format_experiments_summary(experiments_summary)
    formatted_summary.to_csv(formatted_summary_path, index=False)

    if datasets is not None:
        experiments_summary = export_by_dataset(experiments_summary, datasets, experiments_summary_path)
        formatted_summary = export_by_dataset(formatted_summary, datasets, formatted_summary_path)

    return experiments_summary, formatted_summary


def export_by_dataset(experiments_summary: pd.DataFrame,
                      datasets: list[str],
                      path: str
                      ) -> pd.DataFrame:
    if path.endswith(".csv"):
        path = path.replace(".csv", ".xlsx")

    experiments_summary = add_datasets_column(experiments_summary, datasets, index_col="ID")
    with pd.ExcelWriter(path) as writer:
        for dataset in datasets:
            dataset_summary = experiments_summary[experiments_summary["Dataset"] == dataset]
            dataset_summary = dataset_summary.drop("Dataset", axis=1)
            dataset_summary.to_excel(writer, sheet_name=dataset, index=False)

    return experiments_summary


def plot_metric(experiments: dict[str, list[int]],
                results: pd.DataFrame,
                save_folder: str = None,
                metric_name: str = "auroc",
                datasets: list[str] = None,
                use_base_name=True):
    results = add_experiment_id_from_index(experiments, results)
    if use_base_name:
        results["ID"] = results["ID"].apply(os.path.basename)
    else:
        results["ID"] = results["ID"].apply(get_experiment_path)

    if (datasets is not None) and (len(datasets) > 0):
        results = add_datasets_column(results, datasets, index_col="ID")
        results = [results[results["Dataset"] == dataset] for dataset in datasets]
        titles = ["{}_{}".format(dataset, metric_name) for dataset in datasets]
    else:
        results = [results]
        titles = [metric_name]

    for title, dataset_results in zip(titles, results):
        figure = plt.figure(figsize=(10, 7))
        plt.title(title)

        sns.barplot(dataset_results, x="ID", y=metric_name)

        figure.savefig(os.path.join(save_folder, title + ".png"))


def gather_test_results(root: str | Path,
                        summarize_folds: bool,
                        save_absolute_path: bool,
                        save_folder: str = None,
                        datasets: list[str] = None,
                        min_folds: int = -1):
    save_folder = save_folder or root
    if (datasets is not None) and (len(datasets) == 0):
        datasets = None

    test_results_files = find_all_test_results_files(root)
    if len(test_results_files) == 0:
        return

    test_results = load_tests_results(test_results_files)
    test_ids = get_test_ids(test_results_files, save_absolute_path)
    experiments = list_experiments(test_results_files, min_folds)

    gathered_results = pd.concat([test_ids, test_results], axis=1)
    gathered_results.to_csv(os.path.join(save_folder, "gathered_results.csv"), index=False)

    if summarize_folds:
        export_summaries(experiments, test_results, save_absolute_path, save_folder, datasets)

    if "auroc" in test_results:
        # plot_metric(experiments, test_results, save_folder, "auroc", datasets)

        mann_whitney_summary = build_mann_whitney_summary(experiments, test_results)
        mann_whitney_summary.to_csv(os.path.join(save_folder, "mann_whitney_summary.csv"), index=False)


def main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--root", required=True, type=str)
    arg_parser.add_argument("--summarize_folds", default=True, type=bool)
    arg_parser.add_argument("--save_absolute_path", default=False)
    arg_parser.add_argument("--save_folder", default=None, type=str)
    arg_parser.add_argument("--min_folds", default=-1, type=int)
    arg_parser.add_argument("--datasets", nargs="+")

    args = arg_parser.parse_args()

    root: str = args.root
    summarize_folds: bool = args.summarize_folds
    save_absolute_path: bool = args.save_absolute_path in ["True", "true", 1, "Yes", "yes"]
    save_folder: Optional[str] = args.save_folder
    datasets: list[str] = args.datasets
    min_folds: int = args.min_folds or -1

    gather_test_results(root, summarize_folds, save_absolute_path, save_folder, datasets, min_folds)


if __name__ == "__main__":
    main()
