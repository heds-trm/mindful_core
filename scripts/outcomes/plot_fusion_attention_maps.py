import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import os
from pathlib import Path
import argparse
from enum import IntEnum
from typing import Mapping, Hashable, Any

from data.data_folds.data_fold import load_scalar_features, load_categorical_features


class ModalityStatus(IntEnum):
    MISSING = 0
    PARTIAL = 1
    COMPLETE = 2

    def join(self, other: "ModalityStatus"):
        if self == ModalityStatus.MISSING:
            if other == ModalityStatus.MISSING:
                return ModalityStatus.MISSING
            else:
                return ModalityStatus.PARTIAL
        elif self == ModalityStatus.PARTIAL:
            return ModalityStatus.PARTIAL
        else:
            if other == ModalityStatus.COMPLETE:
                return ModalityStatus.COMPLETE
            else:
                return ModalityStatus.PARTIAL

    @property
    def name(self) -> str:
        return str(self).split(".")[-1]


def plot_trial(root: str | Path,
               modalities: list[str],
               modality_statuses: dict[str, ModalityStatus] | None,
               correctness_column: str = None,
               labels_column: str = None
               ) -> None:
    root = Path(root)
    attention_map_paths = search_attention_map_files(root)
    new_module_names = shorten_module_names(list(attention_map_paths.keys()))
    attention_map_paths = {new_module_names[module_name]: path
                           for module_name, path in attention_map_paths.items()}

    test_inferences_path = root / "test_inferences.csv"
    correctness, labels = None, None
    if test_inferences_path.exists():
        test_inferences = pd.read_csv(test_inferences_path, index_col="ScanID")
        if correctness_column in test_inferences:
            correctness = test_inferences[correctness_column]
        if labels_column in test_inferences:
            labels = test_inferences[labels_column]

    figures_folder = root / "fusion_attention"
    figures_folder.mkdir(parents=True, exist_ok=True)

    trial_weights = {}
    attention_width = None
    for module_name, paths in attention_map_paths.items():
        results, all_data = filter_and_plot_module_maps(module_name, paths, modalities,
                                                        modality_statuses, correctness, labels)
        for title, (figure, weights) in results.items():
            if weights is not None:
                trial_weights[title] = weights
                attention_width = len(weights)
            figure.savefig(figures_folder / (title + ".png"))
            plt.close(figure)
        export_attention_map_weights_to_csv(module_name, paths, modalities)
        all_data.to_csv(figures_folder / "{}_trial_data.csv".format(module_name))

    columns = get_columns(attention_width, modalities)
    trial_weights = pd.DataFrame.from_dict(trial_weights, orient="index", columns=columns)
    trial_weights.to_csv(figures_folder / "average_weights.csv")


def shorten_module_names(module_names: list[str]) -> dict[str, str]:
    if len(module_names) == 1:
        return {module_names[0]: module_names[0].split(".")[-1]}

    common_prefix = os.path.commonprefix(module_names)
    return {module_name: module_name[len(common_prefix):]
            for module_name in module_names}


def export_attention_map_weights_to_csv(module_name: str,
                                        attention_map_paths: list[Path],
                                        modalities: list[str],
                                        ):
    ids, attention_maps = load_attention_maps(attention_map_paths)
    columns = get_columns(attention_maps.shape[1], modalities)

    data_frame = pd.DataFrame(attention_maps, columns=columns, index=ids)
    data_frame.index.name = "ScanID"

    export_dir = Path(os.path.commonpath(attention_map_paths), "{}_fusion_weights.csv".format(module_name))
    data_frame.to_csv(export_dir)


def filter_and_plot_module_maps(module_name: str,
                                attention_map_paths: list[Path],
                                modalities: list[str],
                                modality_statuses: dict[str, ModalityStatus] | None,
                                correctness: Mapping[str, bool] | None,
                                labels: Mapping[str, bool] | None
                                ) -> tuple[dict[str, tuple[plt.Figure, np.ndarray]], pd.DataFrame]:
    ids, attention_maps = load_attention_maps(attention_map_paths)

    weights_names = get_columns(attention_maps.shape[1], modalities)
    all_data: dict[str, Any] = {
        weights_names[i]: {_id: float(weights[i]) for _id, weights in zip(ids, attention_maps)}
        for i in range(len(weights_names))
    }
    figure_filters = {module_name: ids}

    if modality_statuses is not None:
        ids_by_status = filter_from_rule(modality_statuses)
        for status, status_ids in ids_by_status.items():
            status: ModalityStatus
            figure_title = module_name + "_" + status.name
            figure_filters[figure_title] = status_ids
        all_data["ModalityStatus"] = modality_statuses

    if correctness is not None:
        ids_by_correctness = filter_from_rule(correctness)
        for figure_title, figure_ids in list(figure_filters.items()):
            figure_ids = np.asarray(figure_ids)
            for correct, correctness_ids in ids_by_correctness.items():
                correctness_label = "Correct" if correct else "Incorrect"
                title = figure_title + "_" + correctness_label
                figure_filters[title] = figure_ids[np.in1d(figure_ids, correctness_ids)]
        all_data["Correctness"] = correctness

    if labels is not None:
        all_data["Label"] = labels

    results = {}
    for figure_title, figure_ids in figure_filters.items():
        figure_attention_maps = filter_attention_maps(attention_maps, ids, figure_ids)
        figure, weights = plot_base_module_maps(figure_title, figure_attention_maps, modalities)
        weights = weights.mean(axis=0)
        results[figure_title] = (figure, weights)

    # region Violin plots
    flat_statuses, flat_correctness = None, None
    if modality_statuses is not None:
        flat_statuses = [modality_statuses[sample_id]
                         if (sample_id in modality_statuses) else ModalityStatus.MISSING
                         for sample_id in ids]
        figure_title = module_name + "_ViolinPlot_ByAvailability"
        figure = plot_by_completeness(figure_title, attention_maps, modalities, flat_statuses)
        results[figure_title] = (figure, None)

    if correctness is not None:
        flat_correctness = [correctness[sample_id] for sample_id in ids]
        figure_title = module_name + "_ViolinPlot_ByCorrectness"
        figure = plot_by_correctness(figure_title, attention_maps, modalities, flat_correctness)
        results[figure_title] = (figure, None)

    if labels is not None:
        flat_labels = [labels[sample_id] for sample_id in ids]
        figure_title = module_name + "_ViolinPlot_ByLabel"
        figure = plot_by_label(figure_title, attention_maps, modalities, flat_labels)
        results[figure_title] = (figure, None)

    if (modality_statuses is not None) and (correctness is not None):
        figure_title = "CorrectnessByAvailability"
        figure = plot_correctness_by_completeness(figure_title, flat_statuses, flat_correctness)
        results[figure_title] = (figure, None)
    # endregion

    data_frame = pd.DataFrame(all_data)
    data_frame.index.name = "ID"

    data_frame["Image to Scalars"] = data_frame["Image"] / (data_frame["Image"] + data_frame["ScalarData"])
    if correctness is not None:
        figure_title = "{}_ImageToScalars_byCorrectness".format(module_name)
        figure = plt.figure(figsize=(10, 7))
        plt.title("{} ({})".format(figure_title, len(attention_maps)))
        sn.barplot(data_frame, y="Image to Scalars", x="Correctness", palette=["#4f81bd", "#c0514d"], hue="Correctness")
        results[figure_title] = (figure, None)

    return results, data_frame


def load_attention_maps(attention_map_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    folds = [np.load(path, allow_pickle=True) for path in attention_map_paths]
    ids = np.concatenate([fold["ids"] for fold in folds], axis=0)
    check_ids_duplicates(ids)
    all_attention_maps = np.concatenate([fold["attention_maps"] for fold in folds], axis=0)

    return ids, all_attention_maps


def filter_from_rule(rule_by_id: Mapping[str, Hashable]) -> dict[Hashable, list[str]]:
    ids_by_rule: dict[Hashable, list[str]] = {}
    for sample_id, rule in rule_by_id.items():
        if rule not in ids_by_rule:
            ids_by_rule[rule] = []
        ids_by_rule[rule].append(sample_id)
    return ids_by_rule


def filter_attention_maps(attention_maps: np.ndarray,
                          base_ids: np.ndarray,
                          filtered_ids: np.ndarray
                          ) -> np.ndarray:
    keep = np.in1d(base_ids, filtered_ids)
    return attention_maps[keep]


def plot_base_module_maps(title: str,
                          attention_maps: np.ndarray,
                          modalities: list[str]) -> tuple[plt.Figure, np.ndarray]:
    mean_attention_map = attention_maps.mean(axis=0, keepdims=attention_maps.ndim == 2)

    index = get_index(mean_attention_map.shape[0], modalities)
    columns = get_columns(mean_attention_map.shape[1], modalities)
    data_frame = pd.DataFrame(mean_attention_map, index=index, columns=columns)

    figure = plt.figure(figsize=(10, 7))
    plt.title("{} ({})".format(title, len(attention_maps)))
    sn.heatmap(data_frame, annot=True, vmin=0.0, vmax=1.0)
    return figure, mean_attention_map


def plot_by_completeness(title: str,
                         attention_maps: np.ndarray,
                         modalities: list[str],
                         modality_statuses: list[ModalityStatus]) -> plt.Figure:
    if attention_maps.ndim > 2:
        attention_maps = attention_maps.mean(axis=1)

    sample_count, modality_count = attention_maps.shape
    attention_weights = attention_maps.reshape([-1])
    modalities = get_columns(attention_maps.shape[1], modalities) * sample_count
    modality_statuses = [status.name
                         for status in modality_statuses
                         for _ in range(modality_count)]

    data_frame = pd.DataFrame(data={"Weight": attention_weights,
                                    "Modality": modalities,
                                    "Availability": modality_statuses})

    figure = plt.figure(figsize=(10, 7))
    plt.title("{} ({})".format(title, len(attention_maps)))
    sn.violinplot(data_frame,
                  x="Modality", y="Weight", hue="Availability",
                  palette=["#9abb59", "#4f81bd", "#c0514d"],
                  native_scale=True, cut=0, density_norm="width")
    return figure


def plot_by_correctness(title: str,
                        attention_maps: np.ndarray,
                        modalities: list[str],
                        correctness: list[bool]) -> plt.Figure:
    if attention_maps.ndim > 2:
        attention_maps = attention_maps.mean(axis=1)

    sample_count, modality_count = attention_maps.shape
    attention_weights = attention_maps.reshape([-1])
    modalities = get_columns(attention_maps.shape[1], modalities) * sample_count
    correctness = ["Correct" if correct else "Incorrect"
                   for correct in correctness
                   for _ in range(modality_count)]

    data_frame = pd.DataFrame(data={"Weight": attention_weights,
                                    "Modality": modalities,
                                    "Correctness": correctness})

    figure = plt.figure(figsize=(10, 7))
    plt.title("{} ({})".format(title, len(attention_maps)))
    sn.violinplot(data_frame,
                  x="Modality", y="Weight", hue="Correctness",
                  palette=["#4f81bd", "#c0514d"],
                  split=True, native_scale=True, cut=0, density_norm="width")
    return figure


def plot_by_label(title: str,
                  attention_maps: np.ndarray,
                  modalities: list[str],
                  labels: list[bool]) -> plt.Figure:
    if attention_maps.ndim > 2:
        attention_maps = attention_maps.mean(axis=1)

    sample_count, modality_count = attention_maps.shape
    attention_weights = attention_maps.reshape([-1])
    modalities = get_columns(attention_maps.shape[1], modalities) * sample_count
    labels = ["DEGEN" if label else "CONTROL"
              for label in labels
              for _ in range(modality_count)]

    data_frame = pd.DataFrame(data={"Weight": attention_weights,
                                    "Modality": modalities,
                                    "Label": labels})

    figure = plt.figure(figsize=(10, 7))
    plt.title("{} ({})".format(title, len(attention_maps)))
    sn.violinplot(data_frame,
                  x="Modality", y="Weight", hue="Label",
                  palette=["#4f81bd", "#c0514d"],
                  split=True, native_scale=True, cut=0, density_norm="width")
    return figure


def plot_correctness_by_completeness(title: str,
                                     modality_statuses: list[ModalityStatus],
                                     correctness: list[bool]) -> plt.Figure:
    order = [status.name for status in reversed(ModalityStatus) if status in modality_statuses]
    modality_statuses = [status.name for status in modality_statuses]
    data_frame = pd.DataFrame(data={"Correctness": correctness,
                                    "Availability": modality_statuses})

    figure = plt.figure(figsize=(5, 7))
    plt.title(title)
    ax = sn.barplot(data_frame,
                    x="Availability", y="Correctness", hue="Availability",
                    order=order, palette=["#9abb59", "#4f81bd", "#c0514d"])

    # correct_for_label / total_for_label

    for i, availability in enumerate(order):
        matching_availability = data_frame[data_frame["Availability"] == availability]
        correct_count = matching_availability["Correctness"].sum()
        matching_count = len(matching_availability)
        ratio = int(round(correct_count / matching_count * 100))
        label = "{} / {} ({}%)".format(correct_count, matching_count, ratio)
        ax.bar_label(ax.containers[i], fontsize=10, labels=[label])

    return figure


def get_index(attention_height: int, modalities: list[str]) -> list[str]:
    if attention_height == len(modalities):
        return modalities
    elif attention_height == 1:
        return ["CLS"]
    elif attention_height == (len(modalities) + 1):
        return ["CLS", *modalities]
    else:
        raise ValueError(attention_height, len(modalities))


def get_columns(attention_width: int, modalities: list[str]) -> list[str]:
    if attention_width == len(modalities):
        return modalities
    elif attention_width == (len(modalities) + 1):
        return ["CLS", *modalities]
    elif attention_width == (len(modalities) + 2):
        exclusive = [modality + "_excl" for modality in modalities]
        return ["CLS", "Mutual", *exclusive]
    else:
        raise ValueError(attention_width, len(modalities))


def check_ids_duplicates(ids: np.ndarray) -> None:
    unique_ids, counts = np.unique(ids, return_counts=True)
    duplicate_ids = unique_ids[counts > 1]
    if len(duplicate_ids):
        raise RuntimeError("Found duplicat ids: {}".format(duplicate_ids))


def search_attention_map_files(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    if root.stem != "lightning_logs":
        root = root / "lightning_logs"

    attention_map_paths: dict[str, list[Path]] = {}
    for path in root.iterdir():
        if path.match("version_*"):
            attention_fold = path / "attention_maps"
            if attention_fold.exists():
                for attention_path in attention_fold.glob("*.npz"):
                    if attention_path.stem not in attention_map_paths:
                        attention_map_paths[attention_path.stem] = []
                    attention_map_paths[attention_path.stem].append(attention_path)

    return attention_map_paths


def expand_features(features: list[str] | None, trials: list[str]) -> list[str | None]:
    if features is None:
        return [None] * len(trials)

    if len(features) != len(trials):
        features = features * (len(trials) // len(features))
        if len(features) != len(trials):
            raise ValueError

    return features


def get_modality_statuses(scalar_path: str | None,
                          categorical_path: str | None
                          ) -> dict[str | int, ModalityStatus] | None:
    if (scalar_path is None) and (categorical_path is None):
        return None

    if scalar_path is not None:
        scalars, _ = load_scalar_features(scalar_path)
        scalars_statuses: dict[str, ModalityStatus] | None = {}
        for sample_id, features in scalars.items():
            if features is None:
                scalars_statuses[sample_id] = ModalityStatus.MISSING
            else:
                missing = [feature is None for feature in features]
                if all(missing):
                    scalars_statuses[sample_id] = ModalityStatus.MISSING
                elif any(missing):
                    scalars_statuses[sample_id] = ModalityStatus.PARTIAL
                else:
                    scalars_statuses[sample_id] = ModalityStatus.COMPLETE
    else:
        scalars_statuses = None

    if categorical_path is not None:
        categorical, categories_names, categories_values = load_categorical_features(categorical_path)
        categorical_statuses: dict[str, ModalityStatus] | None = {}
        for sample_id, features in categorical.items():
            if len(features) == 0:
                categorical_statuses[sample_id] = ModalityStatus.MISSING
            elif len(features) < len(categories_names):
                categorical_statuses[sample_id] = ModalityStatus.PARTIAL
            else:
                categorical_statuses[sample_id] = ModalityStatus.COMPLETE
    else:
        categorical_statuses = None

    statuses: dict[str, ModalityStatus]
    if scalar_path is None:
        statuses = categorical_statuses
    elif categorical_path is None:
        statuses = scalars_statuses
    else:
        statuses = scalars_statuses
        for sample_id, status in categorical_statuses.items():
            if sample_id in statuses:
                statuses[sample_id] = status.join(statuses[sample_id])
            else:
                statuses[sample_id] = status.join(ModalityStatus.MISSING)

    return statuses


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("trials", nargs="+")
    arg_parser.add_argument("--modalities", nargs="+", default=["Image", "Non-image"])
    arg_parser.add_argument("--scalar_features", default=None, nargs="+")
    arg_parser.add_argument("--categorical_features", default=None, nargs="+")
    arg_parser.add_argument("--correctness_column", default="Correctness (EER)")
    arg_parser.add_argument("--labels_column", default="Label")
    args = arg_parser.parse_args()

    trials = args.trials
    scalar_features: list[str] | None = args.scalar_features
    categorical_features: list[str] | None = args.categorical_features

    scalar_features = expand_features(scalar_features, trials)
    categorical_features = expand_features(categorical_features, trials)
    modality_statuses = [get_modality_statuses(scalar_path, categorical_path)
                         for scalar_path, categorical_path in zip(scalar_features, categorical_features)]

    for trial, trial_modality_statuses in zip(trials, modality_statuses):
        plot_trial(trial, args.modalities, trial_modality_statuses, args.correctness_column, args.labels_column)


if __name__ == "__main__":
    main()
