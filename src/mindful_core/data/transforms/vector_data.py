import torch
import numpy as np
from monai.transforms import RandomizableTransform
from monai.utils.type_conversion import convert_data_type, convert_to_tensor, convert_to_dst_type
from monai.data.meta_obj import get_track_meta
from monai.config import DtypeLike
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from copy import deepcopy
from typing import Any

from mindful_core.data import Sample
from mindful_core.data.transforms import SerializableTransform, TransformParameters


# region Scalar data
class ScalarPreprocess(SerializableTransform):
    backend = [TransformBackends.NUMPY]

    def __init__(self, use_standardization=True):
        super(ScalarPreprocess, self).__init__()
        self.use_standardization = use_standardization

        self.mean: np.ndarray | None = None
        self.stddev: np.ndarray | None = None
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None
        self.range: np.ndarray | None = None

    def fit(self, samples: list[Sample]):
        features_count = None
        for i in range(len(samples)):
            if samples[i].scalar_features is not None:
                features_count = len(samples[i].scalar_features)
                break

        if features_count is None:
            raise ValueError("No scalar features were found in samples.")

        mean, stddev = [], []
        _min, _max = [], []
        for feature_index in range(features_count):
            values = [samples[sample_index].scalar_features for sample_index in range(len(samples))]
            values = [value[feature_index] for value in values if (value and value[feature_index])]
            if len(values) > 0:
                values = np.asarray(values, dtype=np.float32)
                mean.append(values.mean())
                stddev.append(values.std())
                _min.append(np.min(values))
                _max.append(np.max(values))
            else:
                # No information about missing modality
                mean.append(0.0)
                stddev.append(1.0)
                _min.append(0.0)
                _max.append(1.0)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.stddev = np.asarray(stddev, dtype=np.float32)
        self.min = np.asarray(_min, dtype=np.float32)
        self.max = np.asarray(_max, dtype=np.float32)
        self.range = self.max - self.min
        # for numerical stability
        self.range = np.where(np.abs(self.range) < 1e-5, np.sign(self.range) * 1e-5, self.range)

    # noinspection PyMethodMayBeStatic
    def requires_fitting(self) -> bool:
        return True

    def __call__(self, features: np.ndarray | None) -> np.ndarray:
        if features is None:
            if self.use_standardization:
                features = np.zeros_like(self.mean)
            else:
                features = (self.mean - self.min) / self.range * 2.0 - 1.0
        else:
            if self.use_standardization:
                features = [self.standardize(features[i], i) for i in range(len(features))]
            else:
                features = [self.normalize(features[i], i) for i in range(len(features))]
            features = np.asarray(features, dtype=np.float32)
        return features

    def normalize(self, feature, feature_index) -> float:
        if feature is None:
            feature = self.mean[feature_index]
        return (feature - self.min[feature_index]) / self.range[feature_index] * 2.0 - 1.0

    def standardize(self, feature, feature_index) -> float:
        if not feature:
            return 0.0

        if self.stddev[feature_index] == 0.0:
            return feature - self.mean[feature_index]

        return (feature - self.mean[feature_index]) / self.stddev[feature_index]

    def reverse(self, scalars: np.ndarray):
        if self.use_standardization:
            return scalars * self.stddev + self.mean
        else:
            return scalars * self.range + self.min

    @classmethod
    def json_identifier(cls) -> str:
        return "scalar_preprocess"

    def to_json(self) -> TransformParameters:
        return {
            "use_standardization": self.use_standardization
        }


class SelectFeatures(SerializableTransform):
    def __init__(self, columns: list[int]):
        super(SelectFeatures, self).__init__()
        self.columns = np.asarray(columns, dtype=int)

    def __call__(self, data: Any):
        return data[self.columns]

    @classmethod
    def json_identifier(cls) -> str:
        return "select_features"

    def to_json(self) -> TransformParameters:
        return {
            "columns": self.columns.tolist()
        }


# endregion

# region Categorical data

class CategoricalPreprocess(SerializableTransform):
    backend = [TransformBackends.NUMPY]

    # TODO: save categories dict for unknown categories for later uses
    def __init__(self,
                 categories: dict[str, dict[Any | None, int]] | list[str],
                 one_hot: bool = False,
                 ):
        super(CategoricalPreprocess, self).__init__()

        categories = deepcopy(categories)
        if isinstance(categories, list):
            categories = {category: {None: 0} for category in categories}
        elif isinstance(categories, dict):
            for category in categories:
                if isinstance(categories[category], list):
                    categories[category] = {label: (i + 1) for i, label in enumerate(categories[category])}

                if 0 in list(categories[category].values()):
                    raise RuntimeError("Could not handle already specified label {} mapped to 0.")
                categories[category][None] = 0

        self.categories: dict[str, dict[Any | None, int]] = categories
        self.one_hot = one_hot

    def fit(self, samples: list[Sample]):
        valid_samples = [sample for sample in samples if sample.categorical_features]
        for sample in valid_samples:
            for key, value in sample.categorical_features.items():
                # if key not in self.categories:
                #   self.categories[key] = {None: 0, value: 1}
                if (key in self.categories) and (value not in self.categories[key]):
                    self.categories[key][value] = len(self.categories[key])

    # noinspection PyMethodMayBeStatic
    def requires_fitting(self) -> bool:
        return True

    def __call__(self, features: dict[str, str] | None) -> np.ndarray:
        # replacing missing features with 0 which means "unknown" / "missing"
        if features is None:
            features = [0 for _ in range(len(self.categories))]
        else:
            features = [self._map_feature(features, key) for key in self.categories]
        features = np.asarray(features, dtype=np.int32)

        if self.one_hot:
            features = self._to_one_hot(features)

        return features

    def _map_feature(self, features: dict[str, str], key: str) -> int:
        if key not in features:
            # Feature value is unknown
            return 0

        feature_value_name = features[key]
        if feature_value_name not in self.categories[key]:
            # Feature value exists but was not in the training set
            return 0

        return self.categories[key][feature_value_name]

    def _to_one_hot(self, features: np.ndarray) -> np.ndarray:
        if features.shape != (len(self.categories),):
            raise ValueError(str(features.shape))

        class_sizes = [len(classes) for classes in self.categories.values()]
        class_indices = np.cumsum([0] + class_sizes)
        total_size = class_indices[-1]
        class_indices = class_indices[:-1]
        features += class_indices

        one_hot = np.zeros(shape=(total_size,), dtype=np.float32)
        one_hot[features] = 1.0

        return one_hot

    def get_categories_class_count(self) -> dict[str, int]:
        return {category: len(classes) for category, classes in self.categories.items()}

    def sort_categories(self, categories: list[str]):
        if len(self.categories) == 0:
            self.categories = {category: {} for category in categories}
        else:
            if (any([category not in self.categories for category in categories]) or
                    any([category not in categories for category in self.categories])):
                raise ValueError("Categories provided did not match registered categories.")
            self.categories = {category: self.categories[category] for category in categories}

    @classmethod
    def json_identifier(cls) -> str:
        return "categorical_preprocess"

    def to_json(self) -> TransformParameters:
        return {
            "categories": self.categories
        }


# endregion

class ConcatenateFeatures(SerializableTransform):
    def __init__(self):
        super(ConcatenateFeatures, self).__init__()

    def __call__(self, *data: Any) -> np.ndarray:
        return np.concatenate(data, axis=-1)

    @classmethod
    def json_identifier(cls) -> str:
        return "concatenate_features"

    # noinspection PyMethodMayBeStatic
    def to_json(self) -> TransformParameters:
        return {}


class RandDropout(RandomizableTransform, SerializableTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self,
                 prob: float = 0.1,
                 feature_prob: float = 0.1,
                 dtype: DtypeLike | str = np.float32,
                 links: list[tuple[float, list[bool]]] = None,
                 ):
        super(RandDropout, self).__init__(prob=prob)
        self.feature_prob = feature_prob
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)
        self.dtype = dtype
        if links is not None:
            self.links = [(prob, np.asarray(link)) for prob, link in links]
        else:
            self.links = None
        self.outcomes: np.ndarray | None = None

    def randomize(self, data: NdarrayOrTensor) -> None:
        super(RandDropout, self).randomize(None)

        if not self._do_transform:
            return

        if self.use_links:
            outcomes_size = len(self.links)
        else:
            outcomes_size = data.shape
        self.outcomes = self.R.uniform(0.0, 1.0, size=outcomes_size)

    def __call__(self, *data: torch.Tensor,
                 randomize: bool = True, **kwargs
                 ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        data = [convert_to_tensor(x, track_meta=get_track_meta()) for x in data]

        if randomize:
            self.randomize(data[0])

        if self._do_transform:
            if self.outcomes is None:
                raise RuntimeError("Please call the `randomize()` function first.")

            if self.use_links:
                mask = np.asarray([True] * len(data[0]))
                for outcome, (prob, link) in zip(self.outcomes, self.links):
                    if outcome <= prob:
                        mask = np.logical_and(mask, link)
            else:
                mask = self.outcomes <= self.feature_prob

            data = [self.apply_mask(x, mask) for x in data]

        if len(data) == 1:
            return data[0]
        return tuple(data)

    def apply_mask(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask, *_ = convert_data_type(mask, dtype=self.dtype)
        tensor, *_ = convert_data_type(tensor, dtype=self.dtype)
        mask, *_ = convert_to_dst_type(mask, tensor)
        return mask * tensor

    @classmethod
    def json_identifier(cls) -> str:
        return "rand_dropout"

    def to_json(self) -> TransformParameters:
        return {
            "prob": self.prob,
            "feature_prob": (self.feature_prob.tolist()
                             if isinstance(self.feature_prob, np.ndarray)
                             else self.feature_prob),
            "dtype": str(self.dtype),
            "links": ([(prob, link.tolist()) for prob, link in self.links]
                      if self.use_links
                      else None)
        }

    @property
    def use_links(self) -> bool:
        return self.links is not None
