import pandas as pd
from deid.config import DeidRecipe
from deid.dicom import get_identifiers, replace_identifiers
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
from typing import Any


def get_dicom_filepaths(root: str | Path, pattern: str) -> dict[str, Path]:
    root = Path(root)

    dicom_filepaths: dict[str, Path] = {}
    for filepath in root.rglob(pattern):
        patient_id = filepath.relative_to(root).parts[0]
        if patient_id in dicom_filepaths:
            print("Patient `{}` already registered:".format(patient_id))
            print("1) Previous path: {}".format(dicom_filepaths[patient_id]))
            print("2) New path: {}".format(filepath))

        dicom_filepaths[patient_id] = filepath

    return dicom_filepaths


def randomize_ids(dictionary: dict[str, Path],
                  seed: int | None = None,
                  prefix: str = ""
                  ) -> tuple[dict[str, Path], pd.DataFrame]:
    new_ids = np.arange(len(dictionary))
    np.random.RandomState(seed).shuffle(new_ids)
    replacements = {old_id: "{}{}".format(prefix, new_id) for old_id, new_id in zip(dictionary.keys(), new_ids)}
    new_id_to_old_path = {new_id: dictionary[old_id] for old_id, new_id in replacements.items()}

    replacements = pd.DataFrame.from_dict(replacements, orient="index")
    replacements.index.name = "PreviousID"
    replacements.columns = ["RandomizedID"]
    return new_id_to_old_path, replacements


def select_dicom_file(dicom_files: list[Path]) -> str:
    dicom_files = filter_dicom_files(dicom_files)
    if len(dicom_files) == 0:
        raise RuntimeError

    return dicom_files[0]


def filter_dicom_files(dicom_files: list[Path]) -> list[str]:
    identifiers = get_identifiers(dicom_files)
    filtered_dicom_files = []
    for dicom_file, dicom_identifiers in identifiers.items():
        number_of_slices = None
        rows = None
        columns = None

        for identifier in dicom_identifiers.values():
            if identifier.name == "NumberOfSlices":
                number_of_slices = identifier.element.value
            elif identifier.name == "Rows":
                rows = identifier.element.value
            elif identifier.name == "Columns":
                columns = identifier.element.value

        if ((number_of_slices is not None) and (number_of_slices > 1)
                and (rows is not None) and (rows == 128)
                and (columns is not None) and (columns == 128)):
            filtered_dicom_files.append(dicom_file)

    return filtered_dicom_files


def normalize_identifiers_paths(base_identifiers: dict[str, Any]) -> dict[str, Any]:
    return {Path(path).as_posix(): _identifiers for path, _identifiers in base_identifiers.items()}


def update_identifiers(dicom_folders: dict[str, Path],
                       tmp_folder: Path,
                       verbose: bool = False,
                       ) -> tuple[dict[str, dict], dict[str, list[str]]]:
    dicom_series: dict[str, list[str]] = {patient_id: [] for patient_id in dicom_folders}
    updated_identifiers: dict[str, dict] = {}

    for patient_id, dicom_folder in tqdm(dicom_folders.items(), desc="Updating identifiers", disable=not verbose):
        dicom_files = list(dicom_folder.glob("*.dcm"))
        dicom_file = select_dicom_file(dicom_files)

        dicom_filename = "{}.dcm".format(patient_id)
        tmp_dicom_file = Path(tmp_folder, dicom_filename).as_posix()

        dicom_series[patient_id].append(dicom_filename)
        shutil.copy(dicom_file, tmp_dicom_file)

        jitter_date = np.random.randint(-31, +31)
        base_identifiers = get_identifiers(tmp_dicom_file)
        base_identifiers = normalize_identifiers_paths(base_identifiers)
        fields = base_identifiers[tmp_dicom_file]
        fields["patient_id"] = patient_id
        fields["source_id"] = patient_id
        fields["jitter_date"] = jitter_date
        updated_identifiers[tmp_dicom_file] = fields

    return updated_identifiers, dicom_series


def anonymize(source_path: Path,
              target_path: Path,
              recipe_path: Path,
              source_pattern: str,
              randomize_patient_ids: bool = False,
              seed: int | None = None,
              verbose: bool = False,
              ) -> list[Path]:
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    recipe = DeidRecipe(recipe_path)

    # region Make temporary and target directories
    tmp_folder = Path(source_path, "tmp")
    if tmp_folder.exists():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir(exist_ok=True)

    target_dicom = Path(target_path, "dicom")
    target_dicom.mkdir(parents=True, exist_ok=True)
    # endregion

    # region Get source dicom folders
    dicom_folders = get_dicom_filepaths(source_path, source_pattern)
    if randomize_patient_ids:
        dicom_folders, replacements = randomize_ids(dicom_folders, seed)
        replacements.to_csv(Path(target_dicom, "replacements.csv"))
    # endregion

    updated_identifiers, dicom_series = update_identifiers(dicom_folders, tmp_folder, verbose=verbose)

    if verbose:
        print("Replacing identifiers...")
    replace_identifiers(dicom_files=list(updated_identifiers.keys()),
                        ids=updated_identifiers,
                        deid=recipe,
                        output_folder=target_dicom,
                        save=True)

    results: list[Path] = []

    progress_bar_description = "Moving deid-ed files from temporary folder to target folder"
    for patient_id, filenames in tqdm(dicom_series.items(), desc=progress_bar_description, disable=not verbose):
        new_patient_folder = Path(target_dicom, patient_id)
        new_patient_folder.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            previous_filepath = Path(target_dicom, filename)
            new_filepath = Path(new_patient_folder, filename)
            results.append(new_filepath)
            shutil.move(previous_filepath, new_filepath)

    shutil.rmtree(tmp_folder)

    return results


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", required=True, type=str)
    arg_parser.add_argument("--target", required=True, type=str)
    arg_parser.add_argument("--recipe", required=True, type=str)
    arg_parser.add_argument("--source_pattern", required=True, type=str)
    arg_parser.add_argument("--randomize_patient_ids", action="store_true")
    arg_parser.add_argument("--seed", default=3930787959, help="Defaults to 3930787959.")

    args = arg_parser.parse_args()
    source_path = Path(args.source)
    target_path = Path(args.target)
    recipe_path = Path(args.recipe)
    source_pattern: str = args.source_pattern
    randomize_patient_ids: bool = args.randomize_patient_ids
    seed = int(args.seed)

    anonymize(source_path, target_path, recipe_path, source_pattern, randomize_patient_ids, seed)


if __name__ == "__main__":
    main()
