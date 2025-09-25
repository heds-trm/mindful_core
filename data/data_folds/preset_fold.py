from pathlib import Path
import json
import pandas as pd
from typing import Union

from mindful_core.data import SubsetID, Sample, Modality, ModalityType
from mindful_core.data.data_folds.data_fold import DataFold


class PresetFold(DataFold):
    def __init__(self,
                 fold_path: str | Path,
                 validation_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 seed: int | None = None,
                 use_abs_path: bool = True,
                 scalar_features_path: str | Path | None = None,
                 categorical_features_path: str | Path | None = None,
                 ):
        self.fold_path: str = fold_path.as_posix() if isinstance(fold_path, Path) else fold_path
        self.use_abs_path = use_abs_path

        super(PresetFold, self).__init__(validation_ratio=validation_ratio,
                                         test_ratio=test_ratio,
                                         seed=seed)

        if scalar_features_path is not None:
            self.load_scalar_features(scalar_features_path)

        if categorical_features_path is not None:
            self.load_categorical_features(categorical_features_path)

    def _init_split(self) -> dict[SubsetID, list[Sample]]:
        return self.train_validation_test_split(self.fold_path)

    def train_validation_test_split(self, fold_path: str = None) -> dict[SubsetID, list[Sample]]:
        if fold_path is None:
            if self.fold_path is None:
                raise ValueError("You must provide `fold_path` when `self.fold_path` is `None`.")
            else:
                fold_path = self.fold_path

        if fold_path.endswith(".json"):
            return self._read_json(fold_path)
        elif fold_path.endswith(".csv"):
            return self._read_csv(fold_path, self.use_abs_path)
        else:
            raise RuntimeError("Invalid fold format: {}.".format(fold_path))

    # region Read the fold from a JSON
    @staticmethod
    def _read_json(fold_path: str):
        with open(fold_path, "r") as file:
            data: dict[str, list[dict[str, Union[str, bool]]]] = json.load(file)

        samples = {
            SubsetID.parse(subset_id): [
                Sample(sample_id=sample["ScanID"],
                       label=sample["Label"],
                       image_path=sample["image_path"])
                for sample in data[subset_id]
            ]
            for subset_id in data
        }

        return samples

    # endregion

    # region Read the fold from a CSV
    @staticmethod
    def _parse_csv_label(label: Union[str, bool, int]) -> int:
        if isinstance(label, bool):
            return int(label)

        if isinstance(label, int):
            return label

        if label.isnumeric():
            return int(label)

        label_map = {
            0: ["false", "control", "healthy", "b"],
            1: ["true", "parkinson", "pd", "patient", "m"],
            2: ["g"]
        }

        label = label.lower()
        for key, values in label_map.items():
            if label in values:
                return key

        raise ValueError("Unknown label value: {}.".format(label))

    def _read_csv(self, fold_path: str, use_abs_path: bool):
        fold_path = Path(fold_path)
        fold_dir = fold_path.parent

        def process_filepath(path: str) -> str:
            if isinstance(path, float):
                # Missing path
                return ""
            
            if (not Path(path).is_absolute()) and use_abs_path:
                path = fold_dir / path
            else:
                path = Path(path)
            return path.as_posix()

        data_frame = pd.read_csv(fold_path)
        if "ScanID" not in data_frame:
            raise RuntimeError("Missing `ScanID` column from sheet at `{}`".format(fold_path))

        # Handle key for images from previous (and deprecated) versions
        if "ScanFilepath" in data_frame:
            data_frame = data_frame.rename(columns={"ScanFilepath": "image:image"})

        image_columns = []
        for column in data_frame:
            if not Modality.is_modality_repr(column):
                continue

            if Modality.parse(column).type == ModalityType.IMAGE:
                data_frame[column] = data_frame[column].apply(process_filepath)
                image_columns.append(column)

        if "Label" in data_frame.columns:
            data_frame["Label"] = data_frame["Label"].apply(self._parse_csv_label)
        record_array = data_frame.to_records(index=False)

        subset_id_map = {subset_id.as_prefix(): subset_id for subset_id in SubsetID}
        dataset = {}
        for record in record_array:
            subset_id = subset_id_map[record["SubsetID"]] if "SubsetID" in data_frame else SubsetID.TRAIN
            label = record["Label"] if "Label" in data_frame.columns else None
            sample = Sample(sample_id=record["ScanID"],
                            label=label,
                            image_path={Modality.parse(column): record[column] for column in image_columns}
                            )

            if subset_id not in dataset:
                dataset[subset_id] = []
            dataset[subset_id].append(sample)

        return dataset

    # endregion
