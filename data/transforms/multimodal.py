import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import DtypeLike
from monai.transforms import RandFlip
from monai.transforms.utils import map_spatial_axes
import numpy as np
from typing import Any, Sequence

from mindful_core.data import ModalityType, Sample, Modality
from mindful_core.data.transforms.serializable_transform import SerializableTransform, TransformParameters
from mindful_core.data.transforms.vector_data import RandDropout


class InferModalityMask(SerializableTransform):
    def __init__(self,
                 modalities: list[str],
                 binary=True,
                 feature_wise=False,
                 scale=1.0,
                 offset=0.0,
                 dtype: DtypeLike | str = None):
        super().__init__()
        if dtype is None:
            dtype = bool if binary else np.float32

        self.modalities: list[ModalityType] = ModalityType.convert_and_check_modalities(modalities)
        self.binary = binary
        self.feature_wise = feature_wise
        self.scale = np.float32(scale)
        self.offset = np.float32(offset)
        self.dtype = np.dtype(dtype) if isinstance(dtype, str) else dtype
        self.templates: list[int | list[str]] | None = None

    def __call__(self, *data: Any) -> np.ndarray:
        if self.feature_wise:
            individual_masks = self.get_feature_wise_masks(*data)
            if len(individual_masks) == 0:
                individual_masks = [[1.0]]
            mask = np.concatenate(individual_masks, axis=0)

        elif self.binary:
            mask = [(not modality_type.is_missing(value)) for modality_type, value in zip(self.modalities, data)]
            mask = np.asarray(mask, dtype=self.dtype)

        else:
            individual_masks = self.get_feature_wise_masks(*data)
            mask = [modality_mask.mean() for modality_mask in individual_masks]
            mask = np.asarray(mask, dtype=self.dtype)

        if self.dtype != bool:
            mask = mask * self.scale + self.offset

        return mask

    def get_feature_wise_masks(self, *data: Any) -> list[np.ndarray]:
        if self.templates is None:
            raise RuntimeError("Make sure you fit the transform/pipeline before you use it.")

        masks = []
        for modality_type, template, value in zip(self.modalities, self.templates, data):
            if value is None:
                if modality_type == ModalityType.CATEGORICAL:
                    template = len(template)
                modality_mask = np.zeros(shape=(template,), dtype=self.dtype)
            else:
                if modality_type == ModalityType.CATEGORICAL:
                    modality_mask = [field in value for field in template]
                elif modality_type == ModalityType.SCALAR:
                    modality_mask = [field is not None for field in value]
                else:
                    modality_mask = [1.0]
                modality_mask = np.asarray(modality_mask, dtype=self.dtype)
            masks.append(modality_mask)

        return masks

    # region Fitting
    # noinspection PyMethodMayBeStatic
    def requires_fitting(self) -> bool:
        return True

    def fit(self, samples: list[Sample]) -> None:
        if (not self.feature_wise) and self.binary:
            return

        self.templates = []
        categories = []
        for modality_type in self.modalities:
            modality = Modality(modality_type)
            if modality_type not in (ModalityType.SCALAR, ModalityType.CATEGORICAL):
                # For other modalities such as images
                self.templates.append(1)
            else:
                for sample in samples:
                    value: np.ndarray | dict[str, str] = sample.get_value(modality)
                    if not modality_type.is_missing(value):
                        if modality_type == ModalityType.SCALAR:
                            self.templates.append(value.shape[0])
                            break
                        elif modality_type == ModalityType.CATEGORICAL:
                            for feature_name in value:
                                if feature_name not in categories:
                                    categories.append(feature_name)

                if modality_type == ModalityType.CATEGORICAL:
                    self.templates.append(categories)

    # endregion

    # region Serialization
    @classmethod
    def json_identifier(cls) -> str:
        return "infer_modality_mask"

    def to_json(self) -> TransformParameters:
        return {
            "modalities": [ModalityType.to_key(modality) for modality in self.modalities],
            "binary": self.binary,
            "feature_wise": self.feature_wise,
            "scale": str(self.scale),
            "offset": str(self.offset),
            "dtype": str(self.dtype),
        }

    @classmethod
    def default_outputs(cls) -> list[str]:
        return [ModalityType.MASK.to_key()]

    # endregion


class RandModalityMask(RandDropout):
    def __init__(self,
                 prob: float = 0.2,
                 feature_prob: float | list[float] = 0.25,
                 min_kept: int = 1
                 ):
        if isinstance(feature_prob, (tuple, list)):
            feature_prob = np.asarray(feature_prob, dtype=np.float32)

        super().__init__(prob, feature_prob, dtype=bool)
        self.min_kept = min_kept

    def randomize(self, data: NdarrayOrTensor) -> None:
        super(RandModalityMask, self).randomize(data)

        if not self._do_transform:
            return

        raise NotImplementedError
        # mask = self.R.uniform(0.0, 1.0, size=data.shape) > self.feature_prob
        # self.mask, *_ = convert_data_type(mask, dtype=self.dtype)

    def __call__(self, *data: Any):
        raise NotImplementedError

    @classmethod
    def json_identifier(cls) -> str:
        return "modality_mask_dropout"

    def to_json(self) -> dict[str, Any]:
        return {
            "prob": self.prob,
            "feature_prob": (self.feature_prob.tolist()
                             if isinstance(self.feature_prob, np.ndarray)
                             else self.feature_prob),
            "dtype": str(self.dtype)
        }


class RandMultimodalFlip(RandFlip, SerializableTransform):
    """
    Currently only supports multiple image modalities if they have the same rank.
    """

    def __init__(self,
                 modalities: list[str],
                 flip_pattern: dict[str, list[str]],
                 prob: float = 0.1,
                 spatial_axis: Sequence[int] | int | None = None,
                 ) -> None:
        super().__init__(prob=prob,
                         spatial_axis=spatial_axis)
        self.modalities: list[ModalityType] = ModalityType.convert_and_check_modalities(modalities)
        if ModalityType.IMAGE not in self.modalities:
            raise ValueError("No image found in expected modalities.")
        self._original_flip_pattern = flip_pattern
        self.flip_pattern: dict[int, list[int]] = {int(axis) + 1: self.parse_pattern(axis_pattern)
                                                   for axis, axis_pattern in flip_pattern.items()}

    def __call__(self, *data: Any, randomize: bool = True) -> tuple:
        if randomize:
            self.randomize(None)

        if not self._do_transform:
            return data

        spatial_axis = None
        for modality_type, value in zip(self.modalities, data):
            if modality_type == ModalityType.IMAGE:
                spatial_axis = map_spatial_axes(value.ndim, spatial_axes=self.flipper.spatial_axis)

        if spatial_axis is None:
            raise RuntimeError("No image found in expected modalities. Did you update `self.modalities` ?")

        flipped_data = []
        for modality_type, value in zip(self.modalities, data):
            if modality_type == ModalityType.IMAGE:
                value = super().__call__(value, randomize=False)
            elif modality_type == ModalityType.SCALAR:
                value = self.flip_scalars(value, spatial_axis)

            flipped_data.append(value)

        return tuple(flipped_data)

    def flip_scalars(self, value: Sequence[float] | None, spatial_axis: list[int]):
        if ModalityType.SCALAR.is_missing(value):
            return value

        for axis in spatial_axis:
            if axis in self.flip_pattern:
                if isinstance(value, (list, tuple)):
                    value = [value[self.flip_pattern[axis][i]] for i in range(len(value))]
                elif isinstance(value, torch.Tensor):
                    value = value[self.flip_pattern[axis]]
                else:
                    raise TypeError(type(value))

        return value

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def json_identifier(cls) -> str:
        return "rand_multimodal_flip"

    def to_json(self) -> TransformParameters:
        return {
            "modalities": [ModalityType.to_key(modality) for modality in self.modalities],
            "flip_pattern": self._original_flip_pattern,
            "prob": self.prob,
            "spatial_axis": self.flipper.spatial_axis,
        }

    @staticmethod
    def parse_pattern(axis_pattern: list[str]) -> list[int]:
        index_map: dict[str, list[int]] = {}

        for index, symbol in enumerate(axis_pattern):
            if symbol not in index_map:
                index_map[symbol] = [index]
            else:
                index_map[symbol].append(index)

        pattern_order = [index_map[symbol].pop(-1) for symbol in axis_pattern]
        return pattern_order
