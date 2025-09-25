import torch
from monai.transforms import LoadImage
from torchmetrics.functional.segmentation import generalized_dice_score
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse


def list_and_filter(folder: Path, prefix: str, suffix: str) -> dict[str, Path]:
    start = len(prefix)
    end = -len(suffix)
    files = {}
    for filepath in folder.iterdir():
        filename = filepath.name
        if filename.startswith(prefix) and filename.endswith(suffix):
            sample_id = filename[start:end]
            files[sample_id] = filepath
    return files


def compute_average_generalized_dice(ground_truth_folder: Path,
                                     ground_truth_prefix: str,
                                     ground_truth_suffix: str,
                                     segmentation_folder: Path,
                                     segmentation_prefix: str,
                                     segmentation_suffix: str,
                                     ) -> pd.DataFrame:
    ground_truth_files = list_and_filter(ground_truth_folder, ground_truth_prefix, ground_truth_suffix)
    segmentation_files = list_and_filter(segmentation_folder, segmentation_prefix, segmentation_suffix)

    loader = LoadImage(ensure_channel_first=True, dtype=torch.int64)

    def load_image(_image_path: str | Path) -> torch.Tensor:
        _image = loader(_image_path)
        # noinspection PyUnresolvedReferences
        _image = (_image == 0).long()
        _image = torch.concat([_image, 1 - _image], dim=0)
        _image = _image.unsqueeze(dim=0)
        return _image

    dice_scores = {}
    for sample_id in tqdm(ground_truth_files):
        if sample_id not in segmentation_files:
            continue

        ground_truth = load_image(ground_truth_files[sample_id])
        segmentation = load_image(segmentation_files[sample_id])

        dice_score = generalized_dice_score(segmentation, ground_truth, num_classes=2, include_background=False)
        dice_scores[sample_id] = float(dice_score)

    dice_scores = pd.DataFrame.from_dict(dice_scores, orient="index", columns=["DICE"])
    dice_scores.index.name = "Sample ID"

    return dice_scores


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--ground_truth_folder")
    arg_parser.add_argument("--ground_truth_prefix")
    arg_parser.add_argument("--ground_truth_suffix")

    arg_parser.add_argument("--segmentation_folder")
    arg_parser.add_argument("--segmentation_prefix")
    arg_parser.add_argument("--segmentation_suffix")

    args = arg_parser.parse_args()
    ground_truth_folder: Path = Path(args.ground_truth_folder)
    ground_truth_prefix: str = args.ground_truth_prefix
    ground_truth_suffix: str = args.ground_truth_suffix

    segmentation_folder: Path = Path(args.segmentation_folder)
    segmentation_prefix: str = args.segmentation_prefix
    segmentation_suffix: str = args.segmentation_suffix

    dice_scores = compute_average_generalized_dice(ground_truth_folder=ground_truth_folder,
                                                   ground_truth_prefix=ground_truth_prefix,
                                                   ground_truth_suffix=ground_truth_suffix,
                                                   segmentation_folder=segmentation_folder,
                                                   segmentation_prefix=segmentation_prefix,
                                                   segmentation_suffix=segmentation_suffix)
    dice_scores.loc["Average"] = dice_scores.mean()

    print(dice_scores)
    dice_scores.to_csv(segmentation_folder / "dice.csv")


if __name__ == "__main__":
    main()
