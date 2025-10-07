import pandas as pd
from pathlib import Path
from shutil import copy2
import argparse

from mindful_core.utils.data_constants import SCAN_ID, DEFAULT_IMAGE_COLUMN
from mindful_core.utils.dataset_preprocessing.anonymize import randomize_ids


def list_ids_in_folder(source: Path,
                       pattern: str = "*"
                       ) -> dict[str, Path]:
    original_files = [filepath for filepath in source.rglob(pattern) if filepath.is_file()]
    original_ids = {filepath.stem: filepath for filepath in original_files}

    return original_ids


def list_ids_from_csv(source: Path) -> dict[str, Path]:
    data_frame = pd.read_csv(source, index_col=SCAN_ID)
    original_ids = data_frame.to_dict()[DEFAULT_IMAGE_COLUMN]
    original_ids = {original_id: Path(original_path) for original_id, original_path in original_ids.items()}
    return original_ids


def replace_ids(source: str | Path,
                destination: str | Path,
                pattern: str = "*",
                prefix: str = "",
                seed: int = None
                ) -> None:
    source = Path(source)
    destination = Path(destination)

    if source.is_dir():
        original_ids = list_ids_in_folder(source, pattern)
    elif source.suffix == ".csv":
        original_ids = list_ids_from_csv(source)
    else:
        raise RuntimeError

    # if existing_replacements_path is not None:
    #     existing_replacements = pd.read_csv(existing_replacements_path, index_col="PreviousID")
    #     existing_replacements = existing_replacements.to_dict()["RandomizedID"]

    new_id_to_old_path, replacements = randomize_ids(original_ids, seed, prefix)

    destination.mkdir(parents=True, exist_ok=True)

    for new_id, old_path in new_id_to_old_path.items():
        if old_path.exists():
            copy_path = destination / "{}{}".format(new_id, old_path.suffix)
            copy2(old_path, copy_path)

    replacements.to_csv(destination / "replacements.csv")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", required=True, type=str)
    arg_parser.add_argument("--destination", required=True, type=str)
    arg_parser.add_argument("--pattern", type=str, default="*")
    arg_parser.add_argument("--prefix", type=str, default="")
    arg_parser.add_argument("--seed", default=3930787959, help="Defaults to 3930787959.")

    args = arg_parser.parse_args()
    source = Path(args.source)
    destination = Path(args.destination)
    pattern: str = args.pattern
    prefix: str = args.prefix
    seed = int(args.seed)

    replace_ids(source, destination, pattern, prefix, seed)


if __name__ == "__main__":
    main()
