import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from typing import Sequence

from data import Sample, SubsetID
from utils.misc import is_defined, write_json
from utils.data_constants import SCAN_ID, LABEL, SUBSET_ID, DEFAULT_IMAGE_COLUMN


class DataFold(object):
    def __init__(self,
                 validation_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 seed: int | None = None,
                 samples: dict[SubsetID, list[Sample]] | None = None,
                 ):
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.samples: dict[SubsetID, list[Sample]] = samples or self._init_split()

        self.scalar_names: list[str] | None = None
        self.categories_names: list[str] | None = None
        self.categories_values: dict[str, list[str | None]] | None = None

    def _init_split(self) -> dict[SubsetID, list[Sample]]:
        raise NotImplementedError("This method must be implemented in subclasses in order to be used.")

    # region Loading (features)
    def load_scalar_features(self, filepath: str):
        features, scalar_names = load_scalar_features(filepath)
        if len(features) == 0:
            features = None
        self.scalar_names = scalar_names
        self.set_scalar_features(features)

    def load_categorical_features(self, filepath: str):
        features, categories_names, categories_values = load_categorical_features(filepath)
        if len(features) == 0:
            features = None
        self.categories_names = categories_names
        self.categories_values = categories_values
        self.set_categorical_features(features)

    # endregion

    # region Set features
    def set_scalar_features(self, scalar_features: dict[str, list[float | None] | None] | None):
        """
        Set the scalar features for each sample in this fold.

        :param scalar_features: A dictionary mapping scan ids to their respective scalar features which are
            lists of optional floats.
        :return: None
        """
        if scalar_features is None:
            return

        for subset_id in self.samples:
            for sample in self.samples[subset_id]:
                if sample.id in scalar_features:
                    sample.scalar_features = scalar_features[sample.id]

    def set_categorical_features(self, categorical_features: dict[str, dict[str, int]]):
        """
        Set the categorical features for each sample in this fold.

        :param categorical_features: A dictionary mapping scan ids to their respective categorical features. Each
            scan id is mapped to a single dictionary itself mapping the id of the category to its value.
        :return: None
        """
        if categorical_features is None:
            return

        for subset_id in self.samples:
            for sample in self.samples[subset_id]:
                if sample.id in categorical_features:
                    sample.categorical_features = categorical_features[sample.id]

    # endregion

    # region Features getters
    @property
    def has_scalar_features(self) -> bool:
        return any([sample.has_scalar_features for sample in self.get_all_samples()])

    @property
    def has_categorical_features(self) -> bool:
        return any([sample.has_categorical_features for sample in self.get_all_samples()])

    # endregion

    @staticmethod
    def extract_columns_groups(data_frame: pd.DataFrame,
                               scan_id_column: str = None,
                               label_column: str = LABEL,
                               subset_column: str = SUBSET_ID,
                               scan_filepath_column: str = DEFAULT_IMAGE_COLUMN
                               ) -> tuple[list[str], list[str], list[str]]:
        if scan_id_column is None:
            scan_id_column = data_frame.index.name

        reserved_columns = [scan_id_column, label_column, subset_column, scan_filepath_column]
        scalar_columns = [str(column_name) for column_name, column_dtype in data_frame.dtypes.items()
                          if column_dtype in [float, np.float32, np.float64]]
        categorical_columns = [str(column_name) for column_name, column_dtype in data_frame.dtypes.items()
                               if (column_dtype in [str, object]) and (column_name not in reserved_columns)]

        return reserved_columns, scalar_columns, categorical_columns

    @staticmethod
    def from_data_frame(data_frame: pd.DataFrame,
                        scan_id_column: str = None,
                        label_column: str = LABEL,
                        subset_column: str = SUBSET_ID,
                        scan_filepath_column: str = DEFAULT_IMAGE_COLUMN,
                        validation_ratio: float = 0.2,
                        test_ratio: float = 0.1,
                        seed: int = None,
                        ) -> "DataFold":
        if scan_id_column is None:
            scan_id_column = data_frame.index.name

        _, scalar_columns, categorical_columns = DataFold.extract_columns_groups(data_frame,
                                                                                 scan_id_column,
                                                                                 label_column,
                                                                                 subset_column,
                                                                                 scan_filepath_column)

        data_frame = data_frame.fillna("")

        samples: dict[SubsetID, list[Sample]] = {subset_id: [] for subset_id in SubsetID}
        for scan_id, scan_data in data_frame.iterrows():
            subset_id = SubsetID.parse(scan_data[subset_column])
            sample = Sample(sample_id=str(scan_id), label=scan_data[label_column],
                            image_path=scan_data[scan_filepath_column],
                            scalar_features=scan_data[scalar_columns],
                            categorical_features=scan_data[categorical_columns])
            samples[subset_id].append(sample)

        for subset_id in SubsetID:
            if len(samples[subset_id]) == 0:
                samples.pop(subset_id)

        return DataFold(samples=samples, seed=seed, validation_ratio=validation_ratio, test_ratio=test_ratio)

    # region Cross-validation
    def make_cross_validation_folds(self,
                                    folds_count: int,
                                    test_partitions_count: int = 1,
                                    validation_partitions_count: int = 1,
                                    ) -> list["DataFold"]:
        """
        Gather all samples from all subsets and split them into `folds_count` balanced partitions.
        From these partitions, `folds_count` folds are built sequentially as follows:
            - For each fold:
                - The first `test_partitions_count` partitions make the test subset
                - The next `validation_partitions_count` partitions make the validation subset
                - The remaining partitions make the train subset

            - Each fold has a different starting partition.

        ### Parameters
        1. folds_count : int
            - The number of folds to generate
        2. test_partitions_count : int
            - The number of partitions in a fold reserved for the test subset
        3. validation_partitions_count : int
            - The number of partitions in a fold reserved for the validation subset

        ### Returns
        - folds : list[DataFold]
            - `folds_count` randomly generated and balanced folds.
        """
        if (folds_count <= 0) or (test_partitions_count < 0) or (validation_partitions_count < 0):
            raise ValueError("Invalid fold count in [{}, {}, {}]."
                             .format(folds_count, test_partitions_count, validation_partitions_count))

        if test_partitions_count >= (folds_count - validation_partitions_count):
            raise ValueError("`test_folds_count` must be less than `folds_count` - `validation_folds_count`, "
                             "but got {} and {} ({} - {})."
                             .format(test_partitions_count, folds_count - validation_partitions_count,
                                     folds_count, validation_partitions_count))

        fold_ratio = 1.0 / folds_count
        samples: list[Sample] = sum(self.samples.values(), [])

        indices_per_label = self.get_indices_per_label(samples, shuffle=True)
        partition_indices_per_label = [np.array_split(label_indices, folds_count)
                                       for label_indices in indices_per_label]
        partition_indices = [np.concatenate([label_partitions_indices[i]
                                             for label_partitions_indices in partition_indices_per_label])
                             for i in range(folds_count)]

        partitions = []
        for partition_indices in partition_indices:
            self.shuffle(partition_indices)
            partition = [samples[index] for index in partition_indices]
            partitions.append(partition)

        folds = []
        for start_index in range(folds_count):
            indices = [(start_index + i) % folds_count for i in range(folds_count)]

            test_fold_indices = indices[:test_partitions_count]
            validation_fold_indices = indices[
                                      test_partitions_count:test_partitions_count + validation_partitions_count]
            train_fold_indices = indices[test_partitions_count + validation_partitions_count:]

            test_folds = sum([partitions[i] for i in test_fold_indices], [])
            validation_folds = sum([partitions[i] for i in validation_fold_indices], [])
            train_folds = sum([partitions[i] for i in train_fold_indices], [])

            fold_samples = {
                SubsetID.TRAIN: train_folds,
            }

            if test_partitions_count > 0:
                fold_samples[SubsetID.TEST] = test_folds

            if validation_partitions_count > 0:
                fold_samples[SubsetID.VALIDATION] = validation_folds

            fold = DataFold(validation_ratio=fold_ratio,
                            test_ratio=fold_ratio,
                            seed=self.seed,
                            samples=fold_samples)
            folds.append(fold)
        return folds

    # endregion

    @staticmethod
    def join_folds(folds: list["DataFold"],
                   validation_ratio: float = 0.2,
                   test_ratio: float = 0.1,
                   seed: int | None = None):
        samples = {}
        for fold in folds:
            for subset_id, subset_samples in fold.samples.items():
                if subset_id not in samples:
                    samples[subset_id] = []
                samples[subset_id] += subset_samples
        return DataFold(validation_ratio=validation_ratio,
                        test_ratio=test_ratio,
                        seed=seed,
                        samples=samples)

    # region Labels
    def get_indices_per_label(self,
                              samples: list[Sample],
                              shuffle: bool,
                              target_ratios: Sequence[float] | None = None,
                              ) -> list[np.ndarray]:
        classes = self.get_classes(samples)
        indices_per_label = [np.asarray([i for i, sample in enumerate(samples)
                                         if sample.label == label], dtype=np.int32)
                             for label in classes]

        if shuffle:
            self.shuffle(*indices_per_label)

        # region Remove excess indices (when target ratios are defined)
        if target_ratios is not None:
            samples_count = len(samples)
            target_count_per_label = [int(samples_count * ratio) for ratio in target_ratios]
            observ_count_per_label = [len(indices) for indices in indices_per_label]

            for label in range(len(target_count_per_label)):
                target_count_near_observed_count = (observ_count_per_label[label] - target_count_per_label[
                    label]) <= 1
                if not target_count_near_observed_count:
                    sample_count = target_count_per_label[label]
                    indices_per_label[label] = indices_per_label[label][:sample_count]
        # endregion

        return indices_per_label

    @staticmethod
    def get_classes(samples: list[Sample]) -> list:
        classes = []
        for sample in samples:
            if sample.label not in classes:
                classes.append(sample.label)
        return classes

    @staticmethod
    def get_class_count(samples: list[Sample]) -> int:
        return len(DataFold.get_classes(samples))

    @staticmethod
    def get_labels_ratios(samples: list[Sample]) -> list[float]:
        counts = {}

        for sample in samples:
            if sample.label not in counts:
                counts[sample.label] = 1
            else:
                counts[sample.label] += 1

        ratios = [counts[label] / len(samples) for label in sorted(counts.keys())]
        return ratios

    @staticmethod
    def get_class_weights(samples: list[Sample]) -> np.ndarray:
        labels = [sample.label for sample in samples]
        classes = np.unique(labels)
        return compute_class_weight("balanced", classes=classes, y=labels)

    # endregion

    # region Export/Saving
    def to_data_frame(self) -> pd.DataFrame:
        if len(self.samples) == 0:
            raise RuntimeError("No known samples to convert to a data frame.")

        values = []
        image_columns = []
        for subset_id, samples in self.samples.items():
            subset_name = subset_id.as_prefix()
            for sample in samples:
                if len(image_columns) == 0:
                    image_columns = [str(modality) for modality in sample.image_path.keys()]
                entry = [sample.id, subset_name, str(sample.label)] + list(sample.image_path.values())
                values.append(entry)

        columns = ["ScanID", "SubsetID", "Label"] + image_columns
        return pd.DataFrame(values, columns=columns)

    def save_scan_data(self, filepath: str | Path):
        if isinstance(filepath, Path):
            filepath = filepath.as_posix()

        if filepath.endswith(".csv"):
            self._save_scan_data_csv(filepath)
        else:
            if not filepath.endswith(".json"):
                filepath += ".json"
            self._save_scan_data_json(filepath)

    def _save_scan_data_csv(self, filepath: str | Path):
        self.to_data_frame().to_csv(filepath, index=False)

    def _save_scan_data_json(self, filepath: str):
        samples = {
            subset_id.as_prefix(): [
                {
                    "ScanID": sample.id,
                    "Label": sample.label,
                    "Images": sample.image_path,
                }
                for sample in self.samples[subset_id]
            ]
            for subset_id in self.samples
        }

        write_json(filepath, samples)

    # endregion

    def shuffle(self, *args: list | np.ndarray):
        for arg in args:
            self.random_state.shuffle(arg)

    @property
    def train_ratio(self) -> float:
        return (1.0 - self.test_ratio) * (1.0 - self.validation_ratio)

    @staticmethod
    def get_scan_id(sample_filename: str) -> str:
        # TODO: depending on Sagemaker, parse this to get patient ID
        return sample_filename

    def get_all_samples(self) -> list[Sample]:
        return sum([subset_samples for subset_samples in self.samples.values()], [])


def load_scalar_features(path: str | Path,
                         ) -> tuple[dict[str, list[float | None] | None], list[str]]:
    data_frame = pd.read_csv(path, index_col=SCAN_ID)

    features = {}
    record_array = data_frame.to_records(index=True)
    for scan_id, *sample_features in record_array:
        scan_id = str(scan_id)
        sample_features = [(value if is_defined(value) else None) for value in sample_features]
        sample_features = sample_features if is_defined(sample_features) else None

        features[scan_id] = sample_features

    scalar_names = list(data_frame.columns)

    return features, scalar_names


def load_categorical_features(path: str | Path,
                              ) -> tuple[dict[str, dict[str, int]], list[str], dict[str, list[str | None]]]:
    data_frame = pd.read_csv(path, index_col=SCAN_ID)

    features = {}
    for scan_id, sample_features in data_frame.iterrows():
        scan_id = str(scan_id)
        sample_features = {column: sample_features[column] for column in data_frame.columns
                           if is_defined(sample_features[column])}
        features[scan_id] = sample_features

    categories_names = list(data_frame.columns)
    categories_values: dict[str, list[str | None]] = {}
    for column in categories_names:
        category_values: list[str | None] = [value for value in data_frame[column].unique() if is_defined(value)]
        categories_values[column] = ["unknown"] + category_values
    return features, categories_names, categories_values
