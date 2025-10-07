import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import copy
from typing import Optional

from mindful_core.utils.data_constants import SCAN_ID, LABEL
from mindful_core.utils.misc import load_json


def gather_spatial_visualizations(trial_path: Path, sample_ids: list[str]) -> None:
    remaining_sample_ids = copy.copy(sample_ids)
    sample_count = len(sample_ids)
    if trial_path.stem != "lightning_logs":
        trial_path = trial_path / "lightning_logs"

    # region Search for visualization folders
    expected_visualizations = {
        "attention_maps": {"RolloutAttention_Center": ["NextToAdditive"]},
        # "saliency": {"SmoothSaliency_Center": ["NextToHSV"]},
        "occlusion_map": {"OcclusionMap_DepthWiseMax": ["NextToAdditive"]},
        # "attention_maps": {"RolloutAttention_Center": ["NextToOverlay", "NextToMul", "NextToHSV", "NextToAdditive"]},
        # "saliency": {"SmoothSaliency_Center": ["NextToOverlay", "NextToHSV"]},
        # "occlusion_maps": {"SmoothSaliency_Center": ["NextToOverlay", "NextToHSV"]},
    }
    trial_versions = [path for path in trial_path.iterdir() if (path.match("*/version_*"))]
    ref_folder = next(iter(expected_visualizations))
    ref_visualization_type = next(iter(expected_visualizations[ref_folder]))
    # endregion

    # region Search for provided ids
    found_sample_ids: dict[Path, list[str]] = {version_path: [] for version_path in trial_versions}
    for version_path in trial_versions:
        ref_path = version_path / ref_folder
        for path in ref_path.iterdir():
            if path.is_file() and path.match("*_{}*".format(ref_visualization_type)):
                sample_id = path.stem.split("_")[0]
                if sample_id in remaining_sample_ids:
                    found_sample_ids[version_path].append(sample_id)
                    remaining_sample_ids.remove(sample_id)
    if len(remaining_sample_ids) > 0:
        raise RuntimeError("Could not find the following ids: {}".format(remaining_sample_ids))
    # endregion

    # region Search for visualization files
    visualization_paths: dict[str, dict[str, Path]] = {}
    for visualization_folder in expected_visualizations:
        for visualization_type in expected_visualizations[visualization_folder]:
            for visualization_render in expected_visualizations[visualization_folder][visualization_type]:
                pattern = "{{}}_{}_{}.png".format(visualization_type, visualization_render)
                if "NextTo" in visualization_render:
                    visualization_render = visualization_render.replace("NextTo", "")

                column = "{}_{}".format(visualization_type, visualization_render)
                visualization_paths[column] = {}

                for version_path, version_samples in found_sample_ids.items():
                    for sample_id in version_samples:
                        filename = pattern.format(sample_id)
                        visualization_paths[column][sample_id] = (version_path / visualization_folder / filename)
    # endregion

    figure, axis_array = plt.subplots(sample_count, len(visualization_paths) + 1)
    figure.subplots_adjust(left=0.05, right=1.0, bottom=0.0, top=0.95, wspace=0.05, hspace=0.08)

    base_done = []
    for j, column in enumerate(visualization_paths):
        for i, sample_id in enumerate(sample_ids):
            visualization_path = visualization_paths[column][sample_id]
            visualization, base_image = load_visualization(visualization_path)
            if (base_image is not None) and (i not in base_done):
                plot_image(axis_array[i, 0], base_image)
                base_done.append(i)
            plot_image(axis_array[i, j + 1], visualization)
        title = column
        title = title.replace("RolloutAttention_Center_", "Attention ")
        title = title.replace("SmoothSaliency_Center_", "Saliency ")
        title = title.replace("OcclusionMap_DepthWiseMax_", "Occlusion ")
        title = title.replace(" HSV", "")
        title = title.replace(" Additive", "")
        title = title.replace(" ", "")
        axis_array[0, j + 1].set_title(title, fontsize=4)

    axis_array[0, 0].set_title("Input", fontsize=4)
    for i in range(sample_count):
        plt.setp(axis_array[i, 0], ylabel=sample_ids[i])

    output_path = trial_path.parent / "spatial_visualizations.png"
    print("Saving figure to {}".format(output_path))
    figure.savefig(output_path, dpi=300)


def plot_image(axis: plt.Axes, image: np.ndarray):
    axis.imshow(image)
    axis.axes.get_xaxis().set_ticks([])
    axis.axes.get_yaxis().set_ticks([])


def load_visualization(visualization_path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    visualization = cv2.imread(visualization_path.as_posix())
    visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
    if "NextTo" in visualization_path.stem:
        split_index = visualization.shape[1] // 2
        base_image = visualization[:, :split_index]
        visualization = visualization[:, split_index:]
    else:
        base_image = None

    return visualization, base_image


def pick_samples(trial_path: Path, count: int) -> list[int]:
    if trial_path.stem == "lightning_logs":
        trial_path = trial_path.parent

    inferences_path = trial_path / "test_inferences.csv"
    if not inferences_path.exists():
        raise FileNotFoundError(inferences_path.as_posix())

    inferences = pd.read_csv(inferences_path.as_posix(), index_col=SCAN_ID)
    error = (inferences[LABEL] - inferences["Probability"]).abs()
    error = error.sort_values()

    failures_count = count // 2
    success_count = count - failures_count

    failures = error[-failures_count:]
    successes = error[:success_count]

    return list(successes.index) + list(failures.index)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_path")
    args = arg_parser.parse_args()

    config_path = args.config_path
    config: dict[str, list[int]] = load_json(config_path)

    for trial, sample_ids in config.items():
        trial = Path(trial)
        if isinstance(sample_ids, int):
            sample_ids = pick_samples(trial, sample_ids)

        sample_ids = [str(sample_id) for sample_id in sample_ids]
        gather_spatial_visualizations(trial, sample_ids)


if __name__ == "__main__":
    main()
