import numpy as np
from pathlib import Path
from typing import Any

from data.modalities import ModalityType, Modality, ModalitySet
from utils.data_constants import SCAN_ID, LABEL, DEFAULT_IMAGE_COLUMN


class Sample(object):
    def __init__(self,
                 sample_id: str,
                 label: int,
                 image_path: str | dict[Modality, str],
                 scalar_features: list[float] = None,
                 categorical_features: dict[str, int] = None,
                 label_name: str = None
                 ):
        label = int(label) if isinstance(label, np.integer) else label
        if isinstance(image_path, str):
            image_path = {Modality(ModalityType.IMAGE): image_path}

        self._check_image_path_types(image_path)

        self.id: str = str(sample_id)
        self.label_name: str | None = label_name

        self.label: int = label
        self.image_path: dict[Modality, str] = image_path
        self.scalar_features: list[float] | None = scalar_features
        self.categorical_features: dict[str, int] | None = categorical_features

    def get_value(self, modality: Modality) -> Any:
        if not isinstance(modality, Modality):
            raise TypeError

        if modality.type == ModalityType.SCALAR:
            return np.asarray(self.scalar_features) \
                if self.scalar_features else None

        elif modality.type == ModalityType.CATEGORICAL:
            return self.categorical_features

        elif modality.type == ModalityType.IMAGE:
            return self.image_path[modality]

        elif modality.type == ModalityType.LABEL:
            return self.label

        elif modality.type == ModalityType.META_DATA:
            raise NotImplementedError

        raise ValueError("`{}` is not a valid modality.".format(modality))

    def get_values(self, modalities: ModalitySet) -> dict[str, Any]:
        values = {}
        for modality in modalities:
            if not isinstance(modality, Modality):
                raise TypeError

            if modality.id in values:
                raise ValueError("Found duplicate modality: `{}`".format(modality))

            values[modality.id] = self.get_value(modality)
        return values

    def __repr__(self):
        label = self.label_name or self.label
        return "{{ID: {sample_id} ({label}) - " \
               "Image path: {image_path} | Features (Scalar: {has_scalar} - " \
               "Categorical: {has_categorical}}}".format(sample_id=self.id,
                                                         label=label,
                                                         image_path=self.image_path,
                                                         has_scalar=self.has_scalar_features,
                                                         has_categorical=self.has_categorical_features)

    def to_tsv_line(self) -> str:
        line = [self.id, self.label, self.image_path]

        if self.has_scalar_features:
            line += self.scalar_features

        if self.has_categorical_features:
            line += list(self.categorical_features.values())

        line = [str(elem) for elem in line]

        return "\t".join(line) + "\n"

    @staticmethod
    def get_tsv_header(samples: list["Sample"]):
        header = [SCAN_ID, LABEL, DEFAULT_IMAGE_COLUMN]

        scalar_examples = [sample.scalar_features for sample in samples if sample.has_scalar_features]
        if len(scalar_examples) > 0:
            header += ["Scalar{}".format(i) for i in range(len(scalar_examples[0]))]

        categorical_examples = [sample.categorical_features for sample in samples if sample.has_categorical_features]
        if len(categorical_examples) > 0:
            header += [key for key in categorical_examples[0]]

        return "\t".join(header) + "\n"

    @property
    def has_scalar_features(self) -> bool:
        return not ModalityType.SCALAR.is_missing(self.scalar_features)

    @property
    def has_categorical_features(self) -> bool:
        return not ModalityType.CATEGORICAL.is_missing(self.categorical_features)

    # region Sanity checks

    @staticmethod
    def _check_image_path_types(image_path) -> None:
        if not isinstance(image_path, dict):
            raise TypeError(type(image_path))

        for key, value in image_path.items():
            if (not isinstance(key, Modality)) or (not isinstance(value, (str, Path))):
                raise TypeError("{} - {}".format(type(key), type(value)))

        for image_modality in image_path:
            if not isinstance(image_modality, Modality):
                raise TypeError("Expected keys of `image_path` to be of type `Modality`.")

    # endregion
