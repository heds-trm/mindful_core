import torch
import torch.nn as nn
import numpy as np
import cv2
from monai.config import DtypeLike, PathLike, NdarrayOrTensor
from monai.transforms import (
    Affine,
    AffineGrid,
    Rotate,

    Resize,
    LoadImage,
    Crop,
    CropForeground,
    ScaleIntensity,

    RandomizableTransform,
    RandSpatialCrop,
    RandGaussianSmooth,
    RandGaussianSharpen,

    Resample
)
from monai.transforms.spatial.functional import flip, affine_func
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode
)
from monai.data import MetaTensor
from monai.data.image_reader import ImageReader
from monai.data.meta_obj import get_track_meta
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_to_tensor
from monai.transforms.utils import map_spatial_axes, create_grid
from pytorch_lightning.utilities.types import STEP_OUTPUT
import matplotlib.cm
from pathlib import Path
import copy
from typing import Sequence, Any, Callable, Literal

from mindful_core.data import Sample
from mindful_core.data.transforms.serializable_transform import SerializableTransform, TransformParameters
from mindful_core.data.transforms.batch_transform import BatchTransform
from mindful_core.utils.imaging import (
    adjust_brightness,
    adjust_contrast,
    adjust_sharpness_3d,
    posterize,
    solarize,
    autocontrast_3d,
    equalize_3d,
    get_3d_image_slices,
    get_3d_gaussian_kernel,
    blur_image_3d_repeat,
    get_center_of_mass,
    spacing_to_affine
)
from mindful_core.utils.tensor_utils import normalize, normal_pdf


# region Image loading
class PreloadImage(LoadImage, SerializableTransform):
    def __init__(self,
                 reader=None,
                 image_only: bool = False,
                 dtype: DtypeLike = np.float32,
                 ensure_channel_first: bool = False,
                 simple_keys: bool = False,
                 prune_meta_pattern: str | None = None,
                 prune_meta_sep: str = ".",
                 *args,
                 **kwargs):
        super(PreloadImage, self).__init__(reader=reader, image_only=image_only, dtype=dtype,
                                           ensure_channel_first=ensure_channel_first, simple_keys=simple_keys,
                                           prune_meta_pattern=prune_meta_pattern, prune_meta_sep=prune_meta_sep,
                                           *args, **kwargs)
        self.cache: dict[str, torch.Tensor] | dict[str, tuple[torch.Tensor, dict]] = {}
        self.reader = reader if isinstance(reader, str) else None

    def preload(self, filenames: Sequence[PathLike]):
        for filename in filenames:
            image_data = super(PreloadImage, self).__call__(filename=filename, reader=None)
            self.cache[filename] = image_data

    def fit(self, samples: list[Sample]):
        raise NotImplementedError
        # filenames = [sample.image_path for sample in samples]
        # self.preload(filenames)

    # noinspection PyMethodMayBeStatic
    def requires_fitting(self) -> bool:
        return True

    def __call__(self, filename: Sequence[PathLike] | PathLike, reader: ImageReader | None = None):
        if filename in self.cache:
            return self.cache[filename]
        else:
            return super(PreloadImage, self).__call__(filename=filename, reader=reader)

    @classmethod
    def json_identifier(cls) -> str:
        return "preload_image"

    def to_json(self) -> dict[str, Any]:
        return {
            "reader": self.reader,
            "image_only": self.image_only,
            "dtype": str(self.dtype),
            "ensure_channel_first": self.ensure_channel_first,
            "simple_keys": self.simple_keys,
            "prune_meta_pattern": self.pattern,
            "prune_meta_sep": self.sep,
        }


class LoadImage4D(LoadImage, SerializableTransform):
    def __init__(self,
                 reader=None,
                 image_only: bool = True,
                 dtype: DtypeLike = np.float32,
                 separate_channel_dim: bool = False,
                 simple_keys: bool = False,
                 prune_meta_pattern: str | None = None,
                 prune_meta_sep: str = ".",
                 filename_pattern: str = "phase_*.mha",
                 *args,
                 **kwargs):
        super(LoadImage4D, self).__init__(reader=reader, image_only=image_only, dtype=dtype,
                                          ensure_channel_first=separate_channel_dim, simple_keys=simple_keys,
                                          prune_meta_pattern=prune_meta_pattern, prune_meta_sep=prune_meta_sep,
                                          *args, **kwargs)
        self.filename_pattern = filename_pattern
        self.reader = reader if isinstance(reader, str) else None

    def __call__(self, folder_path: Sequence[PathLike] | PathLike, reader: ImageReader | None = None):
        filenames = self.get_slices_filenames(Path(folder_path))
        slices = [super(LoadImage4D, self).__call__(filename=filename, reader=reader) for filename in filenames]
        stack_dim = 1 if self.ensure_channel_first else 0
        return torch.stack(slices, dim=stack_dim)

    def get_slices_filenames(self, folder_path: Path) -> list[Path]:
        if not folder_path.is_dir():
            if folder_path.match(self.filename_pattern):
                folder_path = folder_path.parent
            else:
                raise ValueError("Expected filename to be a valid folder, got {}.".format(folder_path))

        filenames = []
        ids = []
        for slice_filename in folder_path.iterdir():
            slice_id = LoadImage4D.get_slice_id(slice_filename)
            if slice_id is not None:
                filenames.append(slice_filename)
                ids.append(slice_id)
        slices_order = np.argsort(ids)
        filenames = [filenames[i] for i in slices_order]

        return filenames

    @staticmethod
    def get_slice_id(filename: Path) -> int | None:
        valid_numbers = [str(i) for i in range(10)]
        numbers = []
        for letter in reversed(filename.stem):
            if letter not in valid_numbers:
                break
            numbers.append(letter)

        if len(numbers) == 0:
            return None

        slice_id = int("".join(reversed(numbers)))
        return slice_id

    @classmethod
    def json_identifier(cls) -> str:
        return "load_image_4d"

    def to_json(self) -> TransformParameters:
        return {
            "reader": self.reader,
            "image_only": self.image_only,
            "dtype": str(self.dtype),
            "ensure_channel_first": self.ensure_channel_first,
            "simple_keys": self.simple_keys,
            "prune_meta_pattern": self.pattern,
            "prune_meta_sep": self.sep,
        }


class LoadImageFallback(SerializableTransform):
    def __init__(self,
                 default_size: list[int],
                 default_spacing: list[float],
                 default_value: float = 0.0,
                 image_only: bool = True,
                 ensure_channel_first: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.default_size = default_size
        self.default_spacing = default_spacing
        self.default_value = default_value

        self.loader = LoadImage(image_only=image_only,
                                ensure_channel_first=ensure_channel_first)

    def __call__(self, filename: str) -> MetaTensor:
        if (len(filename) == 0) or (not Path(filename).exists()):
            # Fallback
            image = torch.full(self.default_size, self.default_value)
            if self.loader.ensure_channel_first:
                image = image.unsqueeze(0)
            return MetaTensor(image, affine=spacing_to_affine(self.default_spacing))
        else:
            return self.loader(filename)

    @classmethod
    def json_identifier(cls) -> str:
        return "load_image_fallback"

    def to_json(self) -> TransformParameters:
        return {
            "default_size": self.default_size,
            "default_spacing": self.default_spacing,
            "default_value": self.default_value,
            "image_only": self.loader.image_only,
            "ensure_channel_first": self.loader.ensure_channel_first,
        }


# endregion

# region Intensity
class StandardizeIntensity(SerializableTransform):
    backend = [TransformBackends.TORCH]

    def __init__(self,
                 channel_wise: bool = False,
                 spatial_dims: int = 3,
                 disable_image: bool = False,
                 print_stats: bool = False,
                 mean: torch.Tensor | list[float] | None = None,
                 stddev: torch.Tensor | list[float] | None = None,
                 ):
        super().__init__()
        self.channel_wise = channel_wise
        self.spatial_dims = spatial_dims

        if mean is not None:
            mean = torch.as_tensor(mean, dtype=torch.float32)

        if stddev is not None:
            stddev = torch.as_tensor(stddev, dtype=torch.float32)

        self.mean: torch.Tensor | list[torch.Tensor] | None = mean
        self.stddev: torch.Tensor | list[torch.Tensor] | None = stddev

        self.disable_image = disable_image
        self.print_stats = print_stats
        self._requires_fitting_preprocessed = (mean is None) or (stddev is None)

    def reset_preprocessed(self) -> None:
        self.mean = []
        self.stddev = []

    def fit_preprocessed(self, sample: torch.Tensor) -> None:
        if self.mean is None:
            self.mean = []
            self.stddev = []

        if self.channel_wise:
            axis = tuple([-i for i in range(1, 1 + self.spatial_dims)])
            sample_mean = sample.mean(axis, keepdim=True)
            sample_stddev = sample.std(axis, keepdim=True)
            self.mean.append(sample_mean)
            self.stddev.append(sample_stddev)
        else:
            self.mean.append(sample.mean())
            self.stddev.append(sample.std())

    def aggregate_preprocessed(self) -> None:
        # channel_wise: [sample_count, channel_count, ...]
        #   otherwise : [sample_count]
        if isinstance(self.mean, list) and isinstance(self.stddev, list):
            means: torch.Tensor = torch.stack(self.mean, dim=0)
            stddevs: torch.Tensor = torch.stack(self.stddev, dim=0)
        else:
            raise RuntimeError("No data available or data was already aggregated.")

        means_var = means.var(dim=0)
        variances = stddevs ** 2
        corrected_stddev = torch.sqrt(means_var + variances.mean(dim=0))
        mean = means.mean(dim=0)

        self.mean = mean
        self.stddev = corrected_stddev

        if self.print_stats:
            print("Computed mean/std: {} ({})".format(mean.squeeze(), corrected_stddev.squeeze()))

    def __call__(self, data: torch.Tensor | MetaTensor) -> torch.Tensor | MetaTensor:
        if self.disable_image:
            return torch.randn_like(data)

        if (self.mean is None) or (self.stddev is None):
            mean = data.mean()
            stddev = data.std()
            return (data - mean) / stddev

        return (data - self.mean) / self.stddev

    @property
    def requires_fitting_preprocessed(self) -> bool:
        return self._requires_fitting_preprocessed

    @classmethod
    def json_identifier(cls) -> str:
        return "standardize_intensity"

    def serialize_stats(self) -> dict[str, float | list[float]]:
        stats = {}
        if self.mean is not None:
            stats["mean"] = self.mean.tolist()

        if self.stddev is not None:
            stats["stddev"] = self.stddev.tolist()

        return stats

    def to_json(self) -> dict[str, Any]:
        return {
            "channel_wise": self.channel_wise,
            "spatial_dims": self.spatial_dims,
            "disable_image": self.disable_image,
            "print_stats": self.print_stats,
            **self.serialize_stats()
        }


class StandardizeIntensityD(SerializableTransform):
    def __init__(self,
                 modalities: list[str],
                 channel_wise: bool = False,
                 spatial_dims: int | list[int] | dict[str, int] = 3,
                 disable_image: bool = False,
                 print_stats: bool = False,
                 means: dict[str, torch.Tensor, list[float]] | None = None,
                 stddevs: dict[str, torch.Tensor, list[float]] | None = None
                 ):
        super().__init__()

        if isinstance(spatial_dims, int):
            spatial_dims = {modality_id: spatial_dims for modality_id in modalities}
        elif isinstance(spatial_dims, (tuple, list)):
            spatial_dims = dict(zip(modalities, spatial_dims))

        means = means or {}
        stddevs = stddevs or {}

        self.modalities = modalities
        self.channel_wise = channel_wise
        self.spatial_dims: dict[str, int] = spatial_dims
        self.disable_image = disable_image
        self.print_stats = print_stats

        self.transforms: dict[str, StandardizeIntensity] = {
            modality_id: StandardizeIntensity(channel_wise,
                                              spatial_dims[modality_id],
                                              disable_image,
                                              print_stats,
                                              mean=means.get(modality_id, None),
                                              stddev=stddevs.get(modality_id, None)
                                              )
            for modality_id in modalities
        }

    def reset_preprocessed(self) -> None:
        for transform in self.transforms.values():
            transform.reset_preprocessed()

    def fit_preprocessed(self, *sample: torch.Tensor | MetaTensor) -> None:
        for modality, transform in zip(sample, self.transforms.values()):
            transform.fit_preprocessed(modality)

    def aggregate_preprocessed(self) -> None:
        for transform in self.transforms.values():
            transform.aggregate_preprocessed()

    def __call__(self, *data: torch.Tensor | MetaTensor) -> list[torch.Tensor | MetaTensor]:
        outputs = [
            self.transforms[modality_id](tensor)
            for modality_id, tensor in zip(self.modalities, data)
        ]
        return outputs

    @property
    def requires_fitting_preprocessed(self) -> bool:
        transforms_requirements = [transform.requires_fitting_preprocessed for transform in self.transforms.values()]
        return any(transforms_requirements)

    @classmethod
    def json_identifier(cls) -> str:
        return "standardize_intensityd"

    def to_json(self) -> TransformParameters:
        stats = {}
        for modality_id, transform in self.transforms.items():
            transform_stats = transform.serialize_stats()
            for stat_id in ["mean", "stddev"]:
                if stat_id in transform_stats:
                    if stat_id not in stats:
                        stats[stat_id] = {}
                    stats[stat_id][modality_id] = transform_stats[stat_id]

        return {
            "modalities": self.modalities,
            "channel_wise": self.channel_wise,
            "spatial_dims": self.spatial_dims,
            "disable_image": self.disable_image,
            "print_stats": self.print_stats,
            **stats
        }



class NormalizeIntensity(SerializableTransform):
    backend = [TransformBackends.TORCH]

    def __init__(self,
                 channel_wise: bool = False,
                 spatial_dims: int = 3,
                 print_stats: bool = False,
                 population_min: torch.Tensor | list[float] | None = None,
                 population_max: torch.Tensor | list[float] | None = None,
                 ):
        super().__init__()
        self.channel_wise = channel_wise
        self.spatial_dims = spatial_dims

        if population_min is not None:
            population_min = torch.as_tensor(population_min, dtype=torch.float32)

        if population_max is not None:
            population_max = torch.as_tensor(population_max, dtype=torch.float32)

        self._min: torch.Tensor | None = population_min
        self._max: torch.Tensor | None = population_max

        self.print_stats = print_stats
        self._requires_fitting_preprocessed = (population_min is None) or (population_max is None)

    def reset_preprocessed(self) -> None:
        self._min = None
        self._max = None

    def fit_preprocessed(self, sample: torch.Tensor) -> None:
        if self._min is None:
            self.reset_preprocessed()

        if self.channel_wise:
            axis = tuple([-i for i in range(1, 1 + self.spatial_dims)])

            sample_min = sample.min(axis, keepdim=True)
            sample_max = sample.max(axis, keepdim=True)
        else:
            sample_min = sample.min()
            sample_max = sample.max()

        if self._min is None:
            self._min = sample_min
            self._max = sample_max
        else:
            self._min = torch.minimum(self._min, sample_min)
            self._max = torch.minimum(self._max, sample_max)

    def aggregate_preprocessed(self) -> None:
        if self.print_stats:
            print("Computed mean/std: {} ({})".format(self._min.squeeze(), self._max.squeeze()))

    def __call__(self, data: torch.Tensor | MetaTensor) -> torch.Tensor | MetaTensor:
        if (self._min is None) or (self._max is None):
            _min, _max = data.min(), data.max()

        else:
            _min, _max = self._min, self._max

        return (data - _min) / (_max - _min)

    @property
    def requires_fitting_preprocessed(self) -> bool:
        return self._requires_fitting_preprocessed

    @classmethod
    def json_identifier(cls) -> str:
        return "normalize_intensity"

    def serialize_stats(self) -> dict[str, float | list[float]]:
        stats = {}
        if self._min is not None:
            stats["population_min"] = self._min.tolist()

        if self._max is not None:
            stats["population_max"] = self._max.tolist()

        return stats

    def to_json(self) -> dict[str, Any]:
        return {
            "channel_wise": self.channel_wise,
            "spatial_dims": self.spatial_dims,
            "print_stats": self.print_stats,
            **self.serialize_stats()
        }


class ClipIntensity(SerializableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self,
                 min_value: float = None,
                 max_value: float = None,
                 percentile: int = None,
                 channel_wise: bool = False,
                 spatial_dims: int = 3):
        super(ClipIntensity, self).__init__()
        if percentile is not None:
            if (min_value is not None) or (max_value is not None):
                raise ValueError("`percentile` is exclusive with `min_value` and `max_value`.")
        elif (min_value is None) and (max_value is None):
            raise ValueError("At least one of `min_value`, `max_value` and `percentile` must be provided.")

        self.percentile = percentile
        self.min_value = min_value
        self.max_value = max_value
        self.channel_wise = channel_wise
        self.spatial_dims = spatial_dims

    def __call__(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:

        if self.percentile is not None:
            if isinstance(data, torch.Tensor):
                min_value = data.quantile(q=self.percentile / 100.0)
                max_value = data.quantile(q=1.0 - self.percentile / 100.0)
            else:
                min_value = np.percentile(data, self.percentile)
                max_value = np.percentile(data, 100 - self.percentile)
        else:
            min_value, max_value = self.min_value, self.max_value

        data = data.clip(min_value, max_value)
        return data

    @classmethod
    def json_identifier(cls) -> str:
        return "clip_intensity"

    def to_json(self) -> dict[str, Any]:
        return {
            "percentile": self.percentile,
            "channel_wise": self.channel_wise,
            "spatial_dims": self.spatial_dims,
        }
    


class CenterIntensityBoost(SerializableTransform):
    backend = [TransformBackends.TORCH]

    def __init__(self,
                 center_radius: int | list[int],
                 boost_power: float | int = 1.0,
                 spatial_dims: int = None,
                 **kwargs):
        super().__init__(**kwargs)

        if spatial_dims is None:
            if isinstance(center_radius, (tuple, list)):
                spatial_dims = len(center_radius)
            else:
                spatial_dims = 3

        if isinstance(center_radius, (tuple, list)):
            if len(center_radius) != spatial_dims:
                raise ValueError("The length of center_radius `{}` must match the number of spatial dims ({})".
                                 format(len(center_radius), spatial_dims))
        else:
            center_radius = [center_radius] * spatial_dims

        self.center_radius = center_radius
        self.boost_power = boost_power
        self.spatial_dims = spatial_dims

    def __call__(self, image: np.ndarray | torch.Tensor | MetaTensor) -> np.ndarray | torch.Tensor | MetaTensor:
        spatial_shape = image.shape[-self.spatial_dims:]
        non_spatial_dims = len(image.shape) - self.spatial_dims
        center_slices = [slice(None)] * non_spatial_dims

        for size, radius in zip(spatial_shape, self.center_radius):
            start, end = size // 2 - radius, size // 2 + radius
            center_slices.append(slice(start, end))

        center = image[center_slices]
        center_min = center.min()
        center_max = center.max()

        above_max = image > center_max
        below_min = image < center_min
        boosted = ~(above_max | below_min)

        min_delta = (image - center_min).abs()
        max_delta = (image - center_max).abs()

        min_offset = center_min - min_delta.pow(self.boost_power)
        max_offset = center_max - max_delta.pow(self.boost_power)

        # noinspection PyUnresolvedReferences
        image = (image * boosted.float()) + (min_offset * below_min.float()) + (max_offset * above_max.float())
        return image

    @classmethod
    def json_identifier(cls) -> str:
        return "center_intensity_boost"

    def to_json(self) -> dict[str, Any]:
        return {
            "center_radius": self.center_radius,
            "boost_power": self.boost_power,
            "spatial_dims": self.spatial_dims,
        }


class PseudoColor(SerializableTransform):
    backend = [TransformBackends.TORCH]

    def __init__(self, color_map: str, **kwargs):
        super().__init__(**kwargs)

        self.color_map = color_map

        color_map_fn = matplotlib.cm.get_cmap(name=color_map, lut=256)
        color_map_array = color_map_fn(range(256))
        color_map_array = color_map_array[:, :3]
        color_map_array = torch.as_tensor(color_map_array)
        self.color_map_array = color_map_array
        self.color_map_by_device = {}

    def __call__(self, image: torch.Tensor | MetaTensor) -> torch.Tensor | MetaTensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected a Tensor, got {}".format(type(image)))

        if torch.is_floating_point(image):
            image = self.convert_image_to_integer(image)

        elif image.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Received an image with incorrect data type: {}".format(image.dtype))

        elif image.max() > 255:
            raise ValueError("Unsupported image range: max={}".format(image.max()))
        
        if image.device not in self.color_map_by_device:
            self.color_map_by_device[image.device] = self.color_map_array.to(image.device)
        color_map_array = self.color_map_by_device[image.device]

        image = image.squeeze()
        dims = len(image.shape)
        image = color_map_array[image]

        permutation = [dims] + list(range(dims))
        image = torch.permute(image, permutation)

        return image

    @staticmethod
    def convert_image_to_integer(image: torch.Tensor | MetaTensor) -> torch.Tensor | MetaTensor:
        image_min = float(image.min())
        image_max = float(image.max())
        if (image_min < 0.0) or (image_max > 255.0):
            raise ValueError("Unsupported image range: min={} and max={}".format(image_min, image_max))
        
        if image_max > 1.0:
            if image_min == image_max:
                raise ValueError("Only one intensity found in image, could not normalize image for colormapping.")
            image = (image - image_min) / (image_max - image_min)

        image = (image * 255).to(torch.int32)

        return image

    @classmethod
    def json_identifier(cls) -> str:
        return "pseudo_color"

    def to_json(self) -> dict[str, Any]:
        return {
            "color_map": self.color_map
        }

# endregion

# region Flipping

class HardFlip(SerializableTransform):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    See `torch.flip` documentation for additional details:
    https://pytorch.org/docs/stable/generated/torch.flip.html

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all the axes
            specified in the tuple.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, spatial_axis: Sequence[int] | int | None = None) -> None:
        super().__init__()
        self.spatial_axis = spatial_axis

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: channel first array, must have shape: (num_channels, H[, W, ..., ])
        """
        axes = map_spatial_axes(image.ndim, self.spatial_axis)
        return torch.flip(image, axes)

    @classmethod
    def json_identifier(cls) -> str:
        return "hard_flip"

    def to_json(self) -> TransformParameters:
        return {
            "spatial_axis": self.spatial_axis
        }


class SelectiveFlip(SerializableTransform):
    """
    Selective version of HardFlip

    """
    backend = [TransformBackends.TORCH]

    def __init__(self, selection: dict[str, list[str]]) -> None:
        super().__init__()
        self.selection: dict[int | tuple[int, ...], list[str]] = {}
        for key, value in selection.items():
            if "," in key:
                axes = tuple([int(axis) for axis in key.split(",")])
            else:
                axes = int(key)
            value = [str(x) for x in value]
            self.selection[axes] = value

    def __call__(self, image: MetaTensor) -> MetaTensor:
        """
        Args:
            image: channel first array, must have shape: (num_channels, H[, W, ..., ])
        """
        axes = self.get_image_flip_axes(image)
        if axes is not None:
            image = torch.flip(image, axes)
        return image

    def get_image_flip_axes(self, image: MetaTensor) -> list[int] | None:
        for spatial_axis, ids in self.selection.items():
            image_path = Path(image.meta["filename_or_obj"])
            if image_path.stem in ids:
                axes = map_spatial_axes(image.ndim, spatial_axis)
                return axes
        return None

    @classmethod
    def json_identifier(cls) -> str:
        return "selective_flip"

    def to_json(self) -> TransformParameters:
        return {
            "selection": self.selection
        }


class AutoFlip(SerializableTransform):
    def __init__(self,
                 intensity_quantile: float = 0.997,
                 spatial_dim: int = 3,
                 ignored_dims: Sequence[int] = None,
                 reversed_dims: Sequence[int] = None,
                 ) -> None:
        super().__init__()
        self.intensity_quantile = intensity_quantile
        self.spatial_dim = spatial_dim
        self.ignored_dims = ignored_dims or []
        self.reversed_dims = reversed_dims or []

    def __call__(self, image: torch.Tensor | MetaTensor) -> torch.Tensor | MetaTensor:
        image_size = image.shape[-self.spatial_dim:]
        center_of_mass = self.get_center_of_mass(image)
        axes = [i + 1 for i, (coord, size)
                in enumerate(zip(center_of_mass, image_size))
                if self.should_flip_axis(i, coord, size)]

        if len(axes) > 0:
            image = torch.flip(image, axes)
        return image

    def get_center_of_mass(self, image: torch.Tensor) -> tuple[int, ...]:
        threshold = torch.quantile(image, q=self.intensity_quantile)
        mask = (image > threshold).to(image.dtype)
        location = get_center_of_mass(image * mask)
        return location[-self.spatial_dim:]

    def should_flip_axis(self, spatial_axis: int, coord: int, size: int) -> bool:
        if spatial_axis in self.ignored_dims:
            return False

        should_flip = coord > (size // 2)
        reverse = spatial_axis in self.reversed_dims
        return should_flip != reverse

    @classmethod
    def json_identifier(cls) -> str:
        return "auto_flip"

    def to_json(self) -> TransformParameters:
        return {
            "intensity_quantile": self.intensity_quantile,
            "spatial_dim": self.spatial_dim,
            "ignored_dims": self.ignored_dims,
            "reversed_dims": self.reversed_dims,
        }


# endregion

# region Cropping

class RandomMultiCrop(RandSpatialCrop, SerializableTransform):
    def __init__(self,
                 roi_size: Sequence[int] | int,
                 crop_count: int,
                 output_size: Sequence[int] | int = None,
                 ):
        super(RandomMultiCrop, self).__init__(roi_size=roi_size,
                                              max_roi_size=None,
                                              random_center=True,
                                              random_size=False)
        self.crop_count = crop_count
        self.output_size = output_size
        self.resize_transform = Resize(output_size) if output_size is not None else None

    def __call__(self,
                 img: torch.Tensor,
                 randomize: bool = True,
                 lazy: bool | None = None
                 ) -> torch.Tensor:  # type: ignore
        crops = [super(RandomMultiCrop, self).__call__(img) for _ in range(self.crop_count)]

        if self.resize_transform is not None:
            crops = [self.resize_transform(crop) for crop in crops]

        if self.crop_count == 1:
            crops = crops[0]

        return crops

    @classmethod
    def json_identifier(cls) -> str:
        return "random_multi_crop"

    def to_json(self) -> dict[str, Any]:
        return {
            "roi_size": self.roi_size,
            "crop_count": self.crop_count,
            "output_size": self.output_size
        }


# region AutoCrop
class AutoDepthCrop(Crop, SerializableTransform):
    def __init__(self,
                 roi_2d_center: Sequence[int] | NdarrayOrTensor | None,
                 roi_2d_size: Sequence[int] | NdarrayOrTensor | None,
                 slice_count: int,
                 depth_dim: int = -1,
                 min_depth=0.0,
                 max_depth=1.0,
                 ):
        super(AutoDepthCrop, self).__init__()
        self.roi_2d_center = roi_2d_center
        self.roi_2d_size = roi_2d_size
        self.roi_2d_slices = self.compute_slices(roi_2d_center, roi_2d_size)
        self.slice_count = slice_count
        self.half_depth = torch.divide(torch.as_tensor(slice_count), 2, rounding_mode="floor")
        self.depth_dim = depth_dim
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, image: torch.Tensor, *args, **kwargs) -> MetaTensor:
        slices = self.get_crop_slices(image)
        image = super(AutoDepthCrop, self).__call__(img=image, slices=slices)
        image = self.add_metadata(image, meta_data={"slices": slices})
        return image

    def get_crop_slices(self, image: torch.Tensor) -> tuple[slice, ...]:
        depth_dim = self.get_depth_dim(image)
        depth_slice = self.get_depth_slice(image)

        slices = list(copy.copy(self.roi_2d_slices))
        slices.insert(depth_dim, depth_slice)

        return tuple(slices)

    def get_depth_slice(self, image: torch.Tensor) -> slice:
        depth_dim = self.get_depth_dim(image)
        depth = image.shape[depth_dim]
        depth_center = self.get_depth_center(image, depth_dim, self.min_depth, self.max_depth)

        depth_start = torch.maximum(depth_center - self.half_depth, torch.as_tensor(0))
        depth_end = torch.minimum(depth_start + self.slice_count, torch.as_tensor(depth))

        depth_slice = slice(int(depth_start), int(depth_end))
        return depth_slice

    def get_depth_dim(self, image: torch.Tensor) -> int:
        image_rank = len(image.shape)
        depth_dim = self.depth_dim if self.depth_dim >= 0 else image_rank + self.depth_dim
        return depth_dim

    @staticmethod
    def get_depth_center(image: torch.Tensor,
                         depth_dim: int = -1,
                         min_depth: float = 0.0,
                         max_depth: float = 1.0,
                         ) -> torch.Tensor:
        if depth_dim < 0:
            depth_dim = len(image.shape) + depth_dim

        histogram_dims = [dim for dim in range(len(image.shape)) if dim != depth_dim]
        depth = image.shape[depth_dim]
        min_depth = int(depth * min_depth)
        max_depth = int(depth * max_depth)

        depth_mean = torch.mean(image, dim=histogram_dims)[min_depth:max_depth]
        depth_center = torch.argmax(depth_mean) + min_depth
        return depth_center

    @classmethod
    def json_identifier(cls) -> str:
        return "auto_depth_crop"

    def to_json(self) -> dict[str, Any]:
        return {
            "roi_2d_center": self.roi_2d_center,
            "roi_2d_size": self.roi_2d_size,
            "slice_count": self.slice_count,
            "depth_dim": self.depth_dim,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth
        }


class AutoCropAndStack(SerializableTransform):
    def __init__(self,
                 roi_2d_center: Sequence[int] | NdarrayOrTensor | None,
                 roi_2d_size: Sequence[int] | NdarrayOrTensor | None,
                 slice_count: int,
                 spatial_size: Sequence[int] = None,
                 scale_intensity: Sequence[float] | bool = None,
                 ):
        super(AutoCropAndStack, self).__init__()
        self.auto_depth_crop = AutoDepthCrop(roi_2d_center=roi_2d_center,
                                             roi_2d_size=roi_2d_size,
                                             slice_count=slice_count)

        self.spatial_size = spatial_size
        spatial_size = [*roi_2d_size, slice_count] if (spatial_size is None) else spatial_size
        self.resize = Resize(spatial_size=spatial_size)

        if isinstance(scale_intensity, bool):
            scale_intensity = [0.0, 1.0] if scale_intensity else None
        self.scale_intensity = scale_intensity
        self.intensity_scaler = ScaleIntensity(*scale_intensity) if (scale_intensity is not None) else None

    def __call__(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        cropped = self.auto_depth_crop(img, *args, **kwargs)
        # if self.spatial_size is not None:
        cropped = self.resize(cropped)
        resized = self.resize(img)

        if self.intensity_scaler is not None:
            cropped = self.intensity_scaler(cropped)
            resized = self.intensity_scaler(resized)

        img = torch.concat([cropped, resized], dim=0)
        return img

    @classmethod
    def json_identifier(cls) -> str:
        return "auto_crop_and_stack"

    def to_json(self) -> dict[str, Any]:
        return {
            "roi_2d_center": self.auto_depth_crop.roi_2d_center,
            "roi_2d_size": self.auto_depth_crop.roi_2d_size,
            "slice_count": self.auto_depth_crop.slice_count,
            "spatial_size": self.spatial_size,
            "scale_intensity": self.scale_intensity
        }


class ScanGlimpse(SerializableTransform):
    def __init__(self, period: int, max_count: int, directory: str,
                 center_method: Literal["center", "max_intensity"] = "center") -> None:
        super().__init__()
        self.period = period
        self.max_count = max_count
        self.counter = 0
        self.directory = Path(directory)
        if not self.directory.exists():
            self.directory.mkdir()
        self.center_method = center_method

    def __call__(self, data: torch.Tensor):
        # shape : [channels, width, height, depth]
        image_id = self.counter // self.period
        if ((self.counter % self.period) == 0) and (image_id < self.max_count):
            data = (data - data.min()) / (data.max() - data.min())
            data = data.cpu().numpy()
            slices = get_3d_image_slices(data, method=self.center_method)
            for dim, sample_slice in enumerate(slices):
                output_path = self.directory / "glimpse_{:02d}_{}.png".format(image_id, dim)
                cv2.imwrite(output_path.as_posix(), sample_slice * 255.0)

        self.counter += 1
        return data

    @classmethod
    def json_identifier(cls) -> str:
        return "scan_glimpse"

    def to_json(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "max_count": self.max_count,
            "directory": self.directory.as_posix(),
        }


class ExperimentalMNICropper(SerializableTransform):
    def __init__(self,
                 quantile: float = None,
                 percent: float = None,
                 blur_count: int = 8,
                 blur_size: int = 7,
                 blur_sigma: float = 0.25):
        super().__init__()
        if (quantile is None) == (percent is None):
            raise ValueError("Either quantile or percent must have a value.")

        self.quantile = quantile
        self.percent = percent
        self.blur_count = blur_count
        self.blur_size = blur_size
        self.blur_sigma = blur_sigma

        self._blur_kernel = get_3d_gaussian_kernel(size=self.blur_size, sigma=self.blur_sigma)

    def __call__(self, image: torch.Tensor, *args, **kwargs) -> MetaTensor:
        diff_of_gaussian = self.compute_diff_of_gaussian(image.cuda()).cpu()
        # threshold = torch.quantile(diff_of_gaussian, q=self.quantile)
        if self.quantile is not None:
            threshold = np.percentile(diff_of_gaussian, q=int(self.quantile * 100))
        else:
            threshold = diff_of_gaussian.max() * self.percent

        def above_threshold(_x):
            return _x > threshold

        cropper = CropForeground(select_fn=above_threshold, return_coords=True)
        diff_of_gaussian, box_start, box_end = cropper(diff_of_gaussian)
        image = cropper.crop_pad(image, box_start, box_end)

        image = self.add_metadata(image, meta_data={"box_start": box_start, "box_end": box_end})
        return image

    def compute_diff_of_gaussian(self, image: torch.Tensor):
        blurred_image = image.pow(2)
        blurred_image_0 = self.blur_image(blurred_image, self.blur_count)
        blurred_image_1 = self.blur_image(blurred_image, self.blur_count - 1)
        return normalize(blurred_image_0 - blurred_image_1)

    def blur_image(self, image: torch.Tensor, n: int) -> torch.Tensor:
        kernel = self._blur_kernel.to(image.device)
        image = blur_image_3d_repeat(image, kernel, n=n)
        return image

    @classmethod
    def json_identifier(cls) -> str:
        return "experimental_mni_cropper"

    def to_json(self) -> TransformParameters:
        return {
            "quantile": self.quantile,
            "blur_count": self.blur_count,
            "blur_size": self.blur_size,
            "blur_sigma": self.blur_sigma,
        }


class CenterOfMassAutoCrop(Crop, SerializableTransform):
    def __init__(self,
                 roi_size: Sequence[int] | NdarrayOrTensor,
                 search_min: Sequence[int | float] | NdarrayOrTensor = None,
                 search_max: Sequence[int | float] | NdarrayOrTensor = None,
                 weights_pow: float = 1.0,
                 ):
        super().__init__()
        self.roi_size = roi_size
        spatial_dim = len(roi_size)
        self.spatial_dim = spatial_dim

        self.search_min = search_min
        self.search_max = search_max
        self.weights_pow = weights_pow

    def __call__(self, image: torch.Tensor, *args, **kwargs) -> MetaTensor:
        slices = self.get_crop_slices(image)
        image = super(CenterOfMassAutoCrop, self).__call__(img=image, slices=slices)
        image = self.add_metadata(image, meta_data={"slices": slices})
        return image

    def get_crop_slices(self, image: torch.Tensor) -> tuple[slice, ...]:
        slices: list[slice] = []
        image_size = image.shape[-self.spatial_dim:]
        center_of_mass = self.get_center_of_mass(image)

        for center, base_size, roi_size in zip(center_of_mass, image_size, self.roi_size):
            half_size = roi_size // 2
            min_center = half_size
            max_center = base_size - (roi_size - half_size)

            center = max(min(max_center, center), min_center)

            start = center - half_size
            stop = start + roi_size
            slices.append(slice(start, stop))

        return tuple(slices)

    def get_center_of_mass(self, image: torch.Tensor | np.ndarray) -> tuple[int, ...]:
        if (self.search_min is not None) or (self.search_max is not None):
            image_size = image.shape[-self.spatial_dim:]
            default_min = [0] * self.spatial_dim

            search_min = RelativeSpatialCrop.convert_search_boundary(self.search_min or default_min, image_size)
            search_max = RelativeSpatialCrop.convert_search_boundary(self.search_max or image_size, image_size)
            search_slices = tuple([slice(start, stop) for start, stop in zip(search_min, search_max)])

            image = super(CenterOfMassAutoCrop, self).__call__(img=image, slices=search_slices)
        else:
            search_min = None

        center_of_mass = get_center_of_mass(image, exponent=self.weights_pow)[-self.spatial_dim:]

        if self.search_min is not None:
            center_of_mass = tuple([x + offset for x, offset in zip(center_of_mass, search_min)])

        return center_of_mass

    @classmethod
    def json_identifier(cls) -> str:
        return "center_of_mass_auto_crop"

    def to_json(self) -> dict[str, Any]:
        return {
            "roi_size": self.roi_size,
            "search_min": self.search_min,
            "search_max": self.search_max,
            "weights_pow": self.weights_pow,
        }


class RelativeSpatialCrop(Crop, SerializableTransform):
    def __init__(
            self,
            roi_center: Sequence[float] | NdarrayOrTensor,
            roi_size: Sequence[float | int] | NdarrayOrTensor,
            lazy: bool = False,
    ) -> None:
        super().__init__(lazy)

        self.roi_center = roi_center
        self.roi_size = roi_size
        self.spatial_dim = len(roi_size)

    def __call__(self, image: torch.Tensor, *args, **kwargs) -> MetaTensor:
        slices = self.get_crop_slices(image)
        image = super(RelativeSpatialCrop, self).__call__(img=image, slices=slices)
        image = self.add_metadata(image, meta_data={"slices": slices})
        return image

    def get_crop_slices(self, image: torch.Tensor) -> tuple[slice, ...]:
        image_size = image.shape[-self.spatial_dim:]
        roi_center = self.convert_search_boundary(self.roi_center, image_size)
        roi_size = self.convert_search_boundary(self.roi_size, image_size)
        return self.compute_slices(roi_center, roi_size)

    @staticmethod
    def convert_search_boundary(value: Sequence[int | float], image_size: Sequence[int]) -> Sequence[int]:
        result = [int(x * size) if isinstance(x, float) else x
                  for (x, size) in zip(value, image_size)]
        return result

    @classmethod
    def json_identifier(cls) -> str:
        return "relative_spatial_crop"

    def to_json(self) -> dict[str, Any]:
        return {
            "roi_center": self.roi_center,
            "roi_size": self.roi_size,
        }


# endregion

# endregion

# region Block extraction
def get_random_block_coordinates(inputs: torch.Tensor,
                                 min_size_ratio,
                                 max_size_ratio,
                                 origin_margin,
                                 spatial_dims: int) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_shape = inputs.shape[-spatial_dims:]

    spatial_shape = torch.as_tensor(spatial_shape, dtype=torch.int32, device=inputs.device)
    spatial_size = spatial_shape.to(torch.float32)

    size_ratio = torch.rand(spatial_dims, device=inputs.device) * (max_size_ratio - min_size_ratio) + min_size_ratio
    block_size = (spatial_size * size_ratio).to(torch.int32).clip(1)
    block_origin = (torch.rand(spatial_dims, device=inputs.device) * (spatial_size - block_size)).to(torch.int32)
    block_origin = block_origin.clip(min=origin_margin)
    block_end = (block_origin + block_size).clip(max=spatial_shape - origin_margin)

    return block_origin, block_end


def replace_block(inputs: torch.Tensor,
                  block_values: torch.Tensor,
                  block_origin: torch.Tensor,
                  block_end: torch.Tensor = None,
                  spatial_dims: int = None) -> torch.Tensor:
    if spatial_dims is None:
        spatial_dims = len(block_values.shape)

    if block_end is None:
        block_end = [block_values.shape[i + 1] - block_origin[i] for i in range(spatial_dims)]

    start = block_origin
    end = block_end
    if spatial_dims == 1:
        inputs[..., start[-1]: end[-1]] = block_values
    elif spatial_dims == 2:
        inputs[..., start[-2]: end[-2], start[-1]: end[-1]] = block_values
    elif spatial_dims == 3:
        inputs[..., start[-3]: end[-3], start[-2]: end[-2], start[-1]: end[-1]] = block_values
    else:
        raise RuntimeError("Spatial dims must be between 1 and 3 (included).")

    return inputs


def random_apply(inputs: torch.Tensor, function: Callable, p: torch.Tensor | float, **kwargs):
    return function(inputs, **kwargs) if torch.rand(1) < p else inputs


def slice_select(x: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, dims: Sequence[int]):
    for start, end, dim in zip(starts, ends, dims):
        indices = torch.arange(start, end, dtype=torch.int32, device=x.device)
        x = torch.index_select(x, dim, indices)
    return x


# endregion

# region Inpainting / Pixel shuffling
class InPaintingBlock(object):
    def __init__(self, origin: torch.Tensor, end: torch.Tensor, values: torch.Tensor):
        self.origin = origin
        self.end = end
        self.values = values

    def __call__(self, inputs: torch.Tensor) -> Any:
        return replace_block(inputs, self.values, self.origin, self.end)


class BatchNoiseInPainting(BatchTransform):
    def __init__(self,
                 inpaint_max_count=1,
                 paint_iter_prob=0.95,
                 inpaint_min_ratio=0.15,
                 inpaint_max_ratio=0.35,
                 ):
        super(BatchNoiseInPainting, self).__init__()
        self.inpaint_max_count = inpaint_max_count
        self.paint_iter_prob = paint_iter_prob
        self.min_ratio = inpaint_min_ratio
        self.max_ratio = inpaint_max_ratio

        self.blocks = []

    # noinspection PyUnusedLocal
    def on_train_batch_start(self, module: nn.Module, batch: Any, batch_idx: int):
        self.reset_blocks()

    def on_train_batch_end(self, module: nn.Module, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        pass

    def reset_blocks(self):
        self.blocks = []

    def make_blocks(self, ref_inputs: torch.Tensor):
        for _ in range(self.inpaint_max_count):
            block_origin, block_end = get_random_block_coordinates(ref_inputs, self.min_ratio,
                                                                   self.max_ratio, origin_margin=3,
                                                                   spatial_dims=len(ref_inputs.shape) - 1)
            block_size = block_end - block_origin
            block_values = torch.randn(block_size.tolist(), device=ref_inputs.device).clip(-1.0, 1.0)

            block = InPaintingBlock(block_origin, block_end, block_values)
            self.blocks.append(block)

            stop_painting = np.random.rand() > self.paint_iter_prob
            if stop_painting:
                break

    def forward(self, inputs) -> torch.Tensor:
        if not self.blocks:
            self.make_blocks(inputs)

        for block in self.blocks:
            inputs = block(inputs)
        return inputs

    def __call__(self, inputs):
        return self.forward(inputs)

    @classmethod
    def json_identifier(cls) -> str:
        return "noise_inpaiting"

    def to_json(self) -> dict[str, Any]:
        return {
            "inpaint_max_count": self.inpaint_max_count,
            "paint_iter_prob": self.paint_iter_prob,
            "min_ratio": self.min_ratio,
            "max_ratio": self.max_ratio,
        }


class RandomNDLocalShuffle(RandomizableTransform, SerializableTransform):
    def __init__(self, prob=0.5, block_min_ratio=0.0, block_max_ratio=0.1):
        RandomizableTransform.__init__(self, prob=prob)
        self.block_min_ratio = block_min_ratio
        self.block_max_ratio = block_max_ratio
        self._block_origin = None
        self._block_end = None
        self._block_values = None
        self._random_indices = None

    def randomize(self, data: Any) -> None:
        RandomizableTransform.randomize(self, data)

        if self._do_transform:
            channels, *spatial_shape = data.shape
            spatial_dims = len(spatial_shape)
            self._block_origin, self._block_end = get_random_block_coordinates(data,
                                                                               self.block_min_ratio,
                                                                               self.block_max_ratio,
                                                                               origin_margin=0,
                                                                               spatial_dims=spatial_dims)
            slice_dims = range(-spatial_dims, 0)
            self._block_values = slice_select(data, self._block_origin, self._block_end, slice_dims).view(channels, -1)
            self._random_indices = torch.randperm(self._block_values.size(-1))

    def __call__(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.randomize(img)

        if not self._do_transform:
            return img

        channels, *spatial_shape = img.shape
        spatial_dims = len(spatial_shape)

        # get block size
        block_size = self._block_end - self._block_origin

        # shuffle block
        block_values = self._block_values[:, self._random_indices]
        block_values = block_values.view(channels, *block_size.tolist())

        # replace original by shuffled
        img = replace_block(img, block_values, self._block_origin, self._block_end, spatial_dims)

        return img

    @classmethod
    def json_identifier(cls) -> str:
        return "random_local_shuffle"

    def to_json(self) -> dict[str, Any]:
        return {
            "prob": self.prob,
            "block_min_ratio": self.block_min_ratio,
            "block_max_ratio": self.block_max_ratio,
        }


# endregion

# region RandAugment3D
class RandAugment3D(RandomizableTransform, SerializableTransform):
    # region Init
    def __init__(self,
                 image_size: tuple[int, int, int],
                 augmentation_ops_count: int = 2,
                 magnitude: float = 0.29,
                 augmentation_filter: list[str] = None):
        RandomizableTransform.__init__(self, prob=1.0)

        self.image_size = image_size
        self.augmentation_ops_count = augmentation_ops_count
        self.magnitude = magnitude
        self.augmentation_filter = augmentation_filter

        self.transforms = {
            "identity": None,

            "shear_x": MirrorShear(factors=self.get_shear_factor(axis=0)),
            "shear_y": MirrorShear(factors=self.get_shear_factor(axis=1)),
            "shear_z": MirrorShear(factors=self.get_shear_factor(axis=2)),

            "translate_x": MirrorTranslate(offset=self.get_translation_offset(axis=0)),
            "translate_y": MirrorTranslate(offset=self.get_translation_offset(axis=1)),
            "translate_z": MirrorTranslate(offset=self.get_translation_offset(axis=2)),

            "rotate_x": MirrorRotate(angle=self.get_rotation_angle(axis=0)),
            "rotate_y": MirrorRotate(angle=self.get_rotation_angle(axis=1)),
            "rotate_z": MirrorRotate(angle=self.get_rotation_angle(axis=2)),

            "brightness": AdjustBrightness(factor=self.adjust_brightness_factor),
            "contrast": AdjustContrast(factor=self.adjust_contrast_factor),
            "sharpness": AdjustSharpness(factor=self.adjust_sharpness_factor),
            "posterization": Posterize(bits=self.posterization_bits),
            "solarization": Solarize(threshold=self.solarization_threshold),
            "auto_contrast": AutoContrast(),
            "equalize": Equalize(),
        }
        self._check_transforms_consistency()

        if self.augmentation_filter is not None:
            self.transforms = {key: value for key, value in self.transforms.items()
                               if key in self.augmentation_filter}

        self._selected_transforms: list[str] | None = None
        self._invert_transforms: list[np.ndarray] | None = None

    # region Augmentation parameters
    def get_shear_factor(self, axis: int
                         ) -> tuple[float, ...]:
        leading_zeros = axis * 2
        ending_zeros = (2 - axis) * 2
        factor = 0.3 * self.magnitude
        shear_params = tuple([0.0] * leading_zeros +
                             [factor] * 2 +
                             [0.0] * ending_zeros)

        return shear_params

    def get_translation_offset(self, axis: int) -> list[float]:
        offsets = [0.0, 0.0, 0.0]
        offsets[axis] = 150.0 / 331.0 * self.image_size[axis]
        return offsets

    def get_rotation_angle(self, axis: int) -> list[float]:
        angles = [0.0, 0.0, 0.0]
        angles[axis] = 30.0 * self.magnitude
        return angles

    @property
    def adjust_brightness_factor(self) -> float:
        return 0.9 * self.magnitude

    @property
    def adjust_contrast_factor(self) -> float:
        return 0.9 * self.magnitude

    @property
    def adjust_sharpness_factor(self) -> float:
        return 0.9 * self.magnitude

    @property
    def posterization_bits(self) -> int:
        return 8 - int(np.round(4 * self.magnitude))

    @property
    def solarization_threshold(self) -> float:
        return 1.0 - self.magnitude

    # endregion
    # endregion

    def randomize(self, image: Any) -> None:
        transforms_keys = list(self.transforms.keys())
        self._selected_transforms = self.R.choice(transforms_keys,
                                                  size=self.augmentation_ops_count,
                                                  replace=False)
        self._invert_transforms = self.R.binomial(n=1, p=0.5, size=self.augmentation_ops_count).astype(bool)

    def __call__(self, image: Any):
        self.randomize(image)

        for transform_key, invert_transform in zip(self._selected_transforms, self._invert_transforms):
            if transform_key == "identity":
                continue

            transform = self.transforms[transform_key]

            if self.transforms_invertability()[transform_key]:
                kwargs = {"invert": invert_transform}
            else:
                kwargs = {}

            image = transform(image, **kwargs)

        return image

    @staticmethod
    def transforms_invertability() -> dict[str, bool]:
        return {
            "identity": False,

            "shear_x": True,
            "shear_y": True,
            "shear_z": True,

            "translate_x": True,
            "translate_y": True,
            "translate_z": True,

            "rotate_x": True,
            "rotate_y": True,
            "rotate_z": True,

            "brightness": True,
            "contrast": True,
            "sharpness": True,

            "posterization": False,
            "solarization": False,
            "auto_contrast": False,
            "equalize": False,
        }

    def _check_transforms_consistency(self) -> None:
        transforms_keys = list(self.transforms.keys())
        invertability_keys = list(self.transforms_invertability().keys())
        unknown_keys = [key for key in transforms_keys + invertability_keys
                        if (key not in transforms_keys) or (key not in invertability_keys)]
        if len(unknown_keys) > 0:
            raise RuntimeError("Inconsistency between available transforms and their "
                               "invertability for the following keys: {}".format(",".join(unknown_keys)))

    # region Serialization
    @classmethod
    def json_identifier(cls) -> str:
        return "rand_augment_3d"

    def to_json(self) -> TransformParameters:
        return {
            "image_size": self.image_size,
            "augmentation_ops_count": self.augmentation_ops_count,
            "magnitude": self.magnitude,
            "augmentation_filter": self.augmentation_filter,
        }
    # endregion


# region Mirror transforms (for RandAugment3D)
class MirrorAffine(Affine):
    def __init__(
            self,
            rotate_params: Sequence[float] | float | None = None,
            shear_params: Sequence[float] | float | None = None,
            translate_params: Sequence[float] | float | None = None,
            scale_params: Sequence[float] | float | None = None,
            affine: NdarrayOrTensor | None = None,
            spatial_size: Sequence[int] | int | None = None,
            mode: str | int = GridSampleMode.BILINEAR,
            padding_mode: str = GridSamplePadMode.REFLECTION,
            normalized: bool = False,
            device: torch.device | None = None,
            dtype: DtypeLike = np.float32,
            image_only: bool = True,
    ) -> None:
        super(MirrorAffine, self).__init__(rotate_params=rotate_params,
                                           shear_params=shear_params,
                                           translate_params=translate_params,
                                           scale_params=scale_params,
                                           affine=affine,
                                           spatial_size=spatial_size,
                                           mode=mode,
                                           padding_mode=padding_mode,
                                           normalized=normalized,
                                           device=device,
                                           dtype=dtype,
                                           image_only=image_only,
                                           )

        self.default_affine_grid = self.affine_grid
        self.mirror_affine_grid = AffineGrid(rotate_params=self.mirror_params(rotate_params),
                                             shear_params=self.mirror_params(shear_params),
                                             translate_params=self.mirror_params(translate_params),
                                             scale_params=self.mirror_params(scale_params),
                                             device=self.affine_grid.device,
                                             dtype=self.affine_grid.dtype,
                                             affine=self.affine_grid.affine
                                             )

    @staticmethod
    def mirror_params(params: Sequence[float] | float | None
                      ) -> Sequence[float] | float | None:
        if params is None:
            return None

        if isinstance(params, float):
            return -params

        return [-param for param in params]

    def __call__(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        if "mirror" in kwargs:
            mirror = kwargs["mirror"]
        elif "invert" in kwargs:
            mirror = kwargs["invert"]
        else:
            mirror = False

        self.affine_grid = self.mirror_affine_grid if mirror else self.default_affine_grid
        return super(MirrorAffine, self).__call__(img)


class MirrorShear(MirrorAffine, SerializableTransform):
    def __init__(self, factors: tuple[float, ...]):
        super(MirrorShear, self).__init__(shear_params=factors,
                                          mode=GridSampleMode.NEAREST,
                                          padding_mode=GridSamplePadMode.ZEROS)

        self.factors = factors

    @classmethod
    def json_identifier(cls) -> str:
        return "mirror_shear"

    def to_json(self) -> TransformParameters:
        return {
            "factors": self.factors
        }


class MirrorTranslate(MirrorAffine, SerializableTransform):
    def __init__(self, offset: Sequence[float]):
        super(MirrorTranslate, self).__init__(translate_params=offset,
                                              mode=GridSampleMode.NEAREST,
                                              padding_mode=GridSamplePadMode.ZEROS)
        self.offset = offset

    @classmethod
    def json_identifier(cls) -> str:
        return "mirror_translate"

    def to_json(self) -> TransformParameters:
        return {
            "offset": self.offset
        }


class MirrorRotate(SerializableTransform):
    def __init__(self, angle: float | Sequence[float]):
        super(MirrorRotate, self).__init__()
        mirror_angle = - angle if isinstance(angle, float) else [-x for x in angle]

        self.angle = angle
        self.default_rotate = self._make_rotate(angle)
        self.mirror_rotate = self._make_rotate(mirror_angle)

    @staticmethod
    def _make_rotate(angle: float | Sequence[float]) -> Rotate:
        return Rotate(angle,
                      keep_size=True,
                      mode=GridSampleMode.NEAREST,
                      padding_mode=GridSamplePadMode.ZEROS)

    def __call__(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        if "mirror" in kwargs:
            mirror = kwargs["mirror"]
        elif "invert" in kwargs:
            mirror = kwargs["invert"]
        else:
            mirror = False

        transform = self.mirror_rotate if mirror else self.default_rotate
        return transform(img)

    @classmethod
    def json_identifier(cls) -> str:
        return "mirror_rotate"

    def to_json(self) -> TransformParameters:
        return {
            "angle": self.angle
        }


# endregion

# region Image adjustments (brightness, contrast, sharpness, contrast, ...)
class AdjustBrightness(SerializableTransform):
    def __init__(self, factor: float):
        super(AdjustBrightness, self).__init__()
        self.factor = factor

    def __call__(self, img: torch.Tensor, invert: bool = False) -> torch.Tensor:
        factor = - self.factor if invert else self.factor
        return adjust_brightness(img, 1.0 + factor)

    @classmethod
    def json_identifier(cls) -> str:
        return "adjust_brightness"

    def to_json(self) -> TransformParameters:
        return {
            "factor": self.factor
        }


class AdjustContrast(SerializableTransform):
    def __init__(self, factor: float):
        super(AdjustContrast, self).__init__()
        self.factor = factor

    def __call__(self, img: torch.Tensor, invert: bool = False) -> torch.Tensor:
        factor = - self.factor if invert else self.factor
        return adjust_contrast(img, 1.0 + factor)

    @classmethod
    def json_identifier(cls) -> str:
        return "adjust_contrast"

    def to_json(self) -> TransformParameters:
        return {
            "factor": self.factor
        }


class AdjustSharpness(SerializableTransform):
    def __init__(self, factor: float):
        super(AdjustSharpness, self).__init__()
        self.factor = factor

    def __call__(self, img: torch.Tensor, invert: bool = False) -> torch.Tensor:
        factor = - self.factor if invert else self.factor
        return adjust_sharpness_3d(img, 1.0 + factor)

    @classmethod
    def json_identifier(cls) -> str:
        return "adjust_sharpness"

    def to_json(self) -> TransformParameters:
        return {
            "factor": self.factor
        }


class Posterize(SerializableTransform):
    def __init__(self, bits: int):
        super(Posterize, self).__init__()
        if not (0 <= bits <= 8):
            raise ValueError("The number if bits should be between 0 and 8. Got {}".format(bits))

        self.bits = bits

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return posterize(img, self.bits)

    @classmethod
    def json_identifier(cls) -> str:
        return "posterize"

    def to_json(self) -> TransformParameters:
        return {
            "bits": self.bits
        }


class Solarize(SerializableTransform):
    def __init__(self, threshold: int | float):
        super(Solarize, self).__init__()

        self.threshold = threshold

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return solarize(img, self.threshold)

    @classmethod
    def json_identifier(cls) -> str:
        return "solarize"

    def to_json(self) -> TransformParameters:
        return {
            "threshold": self.threshold
        }


class AutoContrast(SerializableTransform):
    def __init__(self):
        super(AutoContrast, self).__init__()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return autocontrast_3d(img)

    @classmethod
    def json_identifier(cls) -> str:
        return "autocontrast"

    # noinspection PyMethodMayBeStatic
    def to_json(self) -> TransformParameters:
        return {}


class Equalize(SerializableTransform):
    def __init__(self):
        super(Equalize, self).__init__()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return equalize_3d(img)

    @classmethod
    def json_identifier(cls) -> str:
        return "equalize"

    # noinspection PyMethodMayBeStatic
    def to_json(self) -> TransformParameters:
        return {}


# endregion
# endregion

# region Augmentations 
RandSigmaInterval3D = tuple[float, float] | tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def unpack_rand_sigma_interval_3d(sigma: RandSigmaInterval3D
                                  ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if isinstance(sigma[0], (tuple, list)):
        sigma_x, sigma_y, sigma_z = sigma
    else:
        sigma_x = sigma_y = sigma_z = sigma

    return sigma_x, sigma_y, sigma_z


class RandGaussianSmoothSharpen(RandomizableTransform, SerializableTransform):
    def __init__(self,
                 prob: float = 0.1,
                 smooth_vs_sharpen_prob: float = 0.5,
                 smooth_sigma: RandSigmaInterval3D = (0.25, 1.5),
                 sharpen_sigma_1: RandSigmaInterval3D = (0.5, 1.0),
                 sharpen_sigma_2: float | tuple[float, float, float] | RandSigmaInterval3D = 0.5,
                 sharpen_alpha: tuple[float, float] = (10, 30),
                 approx: Literal["erf", "sampled", "scalespace"] = "erf",
                 ):
        super().__init__(prob)

        self.smooth_vs_sharpen_prob = smooth_vs_sharpen_prob
        self.smooth_sigma = smooth_sigma
        self.sharpen_sigma_1 = sharpen_sigma_1
        self.sharpen_sigma_2 = sharpen_sigma_2
        self.sharpen_alpha = sharpen_alpha
        self.approx = approx

        self._use_smooth: bool | None = None

        smooth_sigma_x, smooth_sigma_y, smooth_sigma_z = unpack_rand_sigma_interval_3d(smooth_sigma)
        sharpen_sigma_1_x, sharpen_sigma_1_y, sharpen_sigma_1_z = unpack_rand_sigma_interval_3d(sharpen_sigma_1)

        if isinstance(sharpen_sigma_2, float):
            sharpen_sigma_2_x = sharpen_sigma_2_y = sharpen_sigma_2_z = sharpen_sigma_2
        elif len(sharpen_sigma_2) == 3:
            sharpen_sigma_2_x, sharpen_sigma_2_y, sharpen_sigma_2_z = sharpen_sigma_2
        else:
            sharpen_sigma_2_x, sharpen_sigma_2_y, sharpen_sigma_2_z = unpack_rand_sigma_interval_3d(sharpen_sigma_2)

        self.rand_smooth = RandGaussianSmooth(prob=1.0,
                                              sigma_x=smooth_sigma_x,
                                              sigma_y=smooth_sigma_y,
                                              sigma_z=smooth_sigma_z,
                                              approx=approx
                                              )

        self.rand_sharpen = RandGaussianSharpen(prob=1.0,
                                                sigma1_x=sharpen_sigma_1_x,
                                                sigma1_y=sharpen_sigma_1_y,
                                                sigma1_z=sharpen_sigma_1_z,
                                                sigma2_x=sharpen_sigma_2_x,
                                                sigma2_y=sharpen_sigma_2_y,
                                                sigma2_z=sharpen_sigma_2_z,
                                                alpha=sharpen_alpha,
                                                approx=approx
                                                )

    def randomize(self, data: Any) -> None:
        super().randomize(None)
        if not self._do_transform:
            return

        self._use_smooth = self.R.rand() < self.smooth_vs_sharpen_prob
        if self._use_smooth:
            self.rand_smooth.randomize(data)
        else:
            self.rand_sharpen.randomize(data)

    def __call__(self, image: NdarrayOrTensor, randomize: bool = True):
        image = convert_to_tensor(image, track_meta=get_track_meta())

        if randomize:
            self.randomize(image)

        if not self._do_transform:
            return image
        elif self._use_smooth:
            return self.rand_smooth(image)
        else:
            return self.rand_sharpen(image)

    # region Serialization
    @classmethod
    def json_identifier(cls) -> str:
        return "rand_gaussian_smooth_sharpen"

    def to_json(self) -> TransformParameters:
        return {
            "prob": self.prob,
            "smooth_vs_sharpen_prob": self.smooth_vs_sharpen_prob,
            "smooth_sigma": self.smooth_sigma,
            "sharpen_sigma_1": self.sharpen_sigma_1,
            "sharpen_sigma_2": self.sharpen_sigma_2,
            "sharpen_alpha": self.sharpen_alpha,
            "approx": self.approx,
        }
    # endregion


class RandSaltPepperNoise(RandomizableTransform, SerializableTransform):
    backend = [TransformBackends.TORCH]

    def __init__(self,
                 prob: float = 0.1,
                 ratio: float = 0.05,
                 pick_opposite: bool = True,
                 ):
        super().__init__(prob)

        self.ratio = ratio
        self.pick_opposite = pick_opposite

        self.mask: torch.Tensor | None = None
        self.noise: torch.Tensor | None = None

    def randomize(self, image: torch.Tensor) -> None:
        super().randomize(image)
        if not self._do_transform:
            return

        # region Get mask        
        mask = self.R.rand(*image.shape) < self.ratio
        self.mask = torch.as_tensor(mask, dtype=image.dtype, device=image.device)
        # endregion

        # region Get noise
        if self.pick_opposite:
            use_min = image < image.median()
        else:
            use_min = self.R.rand(*image.shape) < 0.5
            use_min = torch.as_tensor(use_min, dtype=torch.bool, device=image.device)

        scale = float(image.std().cpu()) * 5e-2
        small_noise = self.R.normal(loc=0.0, scale=scale, size=image.shape)
        small_noise = torch.as_tensor(small_noise, dtype=image.dtype, device=image.device)

        min_noise, max_noise = image.min(), image.max()
        noise = torch.where(use_min, min_noise, max_noise)
        noise = (noise + small_noise).clip(min_noise, max_noise)
        self.noise = noise
        # endregion

    def __call__(self, image: NdarrayOrTensor, randomize: bool = True):
        image = convert_to_tensor(image, track_meta=get_track_meta())

        if randomize:
            self.randomize(image)

        if not self._do_transform:
            return image

        if (self.noise is None) or (self.mask is None):
            raise RuntimeError("Please call the `randomize()` function first.")

        image = (self.noise * self.mask) + (image * (1.0 - self.mask))
        return image

    # region Serialization
    @classmethod
    def json_identifier(cls) -> str:
        return "rand_salt_pepper_noise"

    def to_json(self) -> TransformParameters:
        return {
            "prob": self.prob,
            "ratio": self.ratio,
            "pick_opposite": self.pick_opposite,
        }
    # endregion


class RandChannelDropout(RandomizableTransform, SerializableTransform):
    backend = [TransformBackends.TORCH]

    def __init__(self,
                 prob: float = 0.5,
                 channels_dim: int = 0,
                 rescale: bool = True,
                 ):
        super().__init__(prob)

        self.channels_dim = channels_dim
        self.rescale = rescale
        self.mask: torch.Tensor | None = None

    def randomize(self, image: torch.Tensor) -> None:
        super().randomize(image)
        if not self._do_transform:
            return

        channels = image.shape[self.channels_dim]
        dropped_count = torch.randint(low=1, high=channels, size=())

        mask_shape = [1] * len(image.shape)
        mask_shape[self.channels_dim] = channels

        mask_noise = torch.rand(channels, device=image.device)
        # noinspection PyTypeChecker
        top_k, _ = torch.topk(mask_noise, k=dropped_count)

        mask = (mask_noise < top_k).to(image.dtype)
        self.mask = torch.reshape(mask, mask_shape)

    def __call__(self, image: NdarrayOrTensor, randomize: bool = True):
        image = convert_to_tensor(image, track_meta=get_track_meta())

        if randomize:
            self.randomize(image)

        if not self._do_transform:
            return image

        image = image * self.mask
        if self.rescale:
            ratio = self.mask.numel() / self.mask.sum()
            image *= ratio

        return image

    # region Serialization
    @classmethod
    def json_identifier(cls) -> str:
        return "rand_channel_dropout"

    def to_json(self) -> TransformParameters:
        return {
            "prob": self.prob,
            "channels_dim": self.channels_dim,
            "rescale": self.rescale
        }
    # endregion


class RandSyncedNDAugment(RandomizableTransform, SerializableTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self,
                 modalities_spatial_dims: list[list[int]],
                 prob: float = 1.0,

                 flip_prob: float = 0.5,

                 rotate_prob: float = 0.5,
                 rotate_range: float = 3.1416,

                 translate_prob: float = 0.5,
                 translate_range: int | list[int] = 8,

                 affine_padding: str = "zeros",
                 device: str = "cpu"
                 ):
        super().__init__(prob)

        self.modalities_spatial_dims = modalities_spatial_dims
        self.flip_prob = flip_prob

        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.affine_padding = affine_padding

        self.translate_prob = translate_prob
        self.translate_range = translate_range

        self.device = device

        self._max_spatial_dim = max([max(modality_spatial_dims) for modality_spatial_dims in modalities_spatial_dims])
        self._probs = np.asarray([flip_prob, rotate_prob, translate_prob])

        if not isinstance(translate_range, (tuple, list)):
            translate_range = [translate_range] * self._max_spatial_dim
        self._translate_range = np.asarray(translate_range, dtype=np.float32)
        self._resampler = Resample(device)

        self.do_flip = None
        self.flip_dims = None

        self.do_rotate = None
        self.rotate_angles = None

        self.do_translate = None
        self.translate_offset = None

    def randomize(self, *images: torch.Tensor | MetaTensor) -> None:
        super().randomize(images)
        if not self._do_transform:
            return

        do_transforms = self.R.uniform(size=len(self._probs)) < self._probs
        if not np.any(do_transforms):
            self._do_transform = False
            return

        self.do_flip = do_transforms[0]
        if self.do_flip:
            flip_dims = self.R.permutation(self._max_spatial_dim)
            flip_dims_count = self.R.randint(self._max_spatial_dim) + 1
            self.flip_dims = flip_dims[:flip_dims_count]
        else:
            self.flip_dims = None

        self.do_rotate = do_transforms[1]
        if self.do_rotate:
            self.rotate_angles = self.R.uniform(low=-self.rotate_range,
                                                high=self.rotate_range,
                                                size=[self._max_spatial_dim + 1])
        else:
            self.rotate_angles = np.zeros(shape=[self._max_spatial_dim + 1])

        self.do_translate = do_transforms[2]
        if self.do_translate:
            self.translate_offset = self.R.uniform(low=-self.translate_range,
                                                   high=self.translate_range,
                                                   size=[self._max_spatial_dim + 1])
        else:
            self.translate_offset = np.zeros(shape=[self._max_spatial_dim + 1])

        if (not self.do_flip) and (not self.do_rotate) and (not self.do_translate):
            self._do_transform = False

    def __call__(self, *images: torch.Tensor | MetaTensor, randomize: bool = True) -> list[torch.Tensor | MetaTensor]:
        if randomize:
            self.randomize(*images)

        if not self._do_transform:
            return list(images)

        images = [convert_to_tensor(image, track_meta=get_track_meta()) for image in images]

        if self.do_flip:
            images = [self.flip_image(image, modality_spatial_dims)
                      for image, modality_spatial_dims
                      in zip(images, self.modalities_spatial_dims)]

        if self.do_affine:
            images = [self.affine_image(image, modality_spatial_dims)
                      for image, modality_spatial_dims
                      in zip(images, self.modalities_spatial_dims)]

        return images

    def flip_image(self, image: torch.Tensor | MetaTensor,
                   modality_spatial_dims: list[int]) -> torch.Tensor | MetaTensor:
        flip_dims = [dim for dim in self.flip_dims if dim in modality_spatial_dims]
        return flip(image, flip_dims, lazy=False, transform_info={})

    def affine_image(self, image: torch.Tensor | MetaTensor,
                     modality_spatial_dims: list[int]) -> torch.Tensor | MetaTensor:
        rotate_params = [self.rotate_angles[i] for i in modality_spatial_dims]
        translate_params = [self.translate_offset[i] for i in modality_spatial_dims]
        spatial_size = image.shape[1:]

        # noinspection PyTypeChecker
        affine_grid = AffineGrid(
            rotate_params=rotate_params,
            translate_params=translate_params,
            device=self.device,
            dtype=image.dtype,
            lazy=False,
        )

        # TODO: cache 1 grid per input modality
        # noinspection PyTypeChecker
        grid = create_grid(spatial_size, device=self.device, backend="torch")
        affine = affine_grid(spatial_size, grid)[1]
        image = affine_func(image, affine, grid,
                            self._resampler, spatial_size,
                            mode="bilinear",
                            padding_mode=self.affine_padding,
                            do_resampling=True,
                            image_only=True,
                            lazy=False,
                            transform_info={})
        return image

    @property
    def do_affine(self) -> bool:
        return self.do_rotate or self.do_translate

    # region Serialization
    @classmethod
    def json_identifier(cls) -> str:
        return "rand_synced_nd_augment"

    def to_json(self) -> TransformParameters:
        return {
            "modalities_spatial_dims": self.modalities_spatial_dims,
            "prob": self.prob,

            "flip_prob": self.flip_prob,

            "rotate_prob": self.rotate_prob,
            "rotate_range": self.rotate_range,

            "translate_prob": self.translate_prob,
            "translate_range": self.translate_range,

            "affine_padding": self.affine_padding,
            "device": self.device
        }
    # endregion


# endregion

# region Global position embedding

class ComputeGlobalPositionEmbedding(SerializableTransform):
    def __init__(self,
                 scale: float,
                 spatial_dims: int = 3,
                 device: str = "cpu",
                 **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.spatial_dims = spatial_dims
        self.device = device

    def __call__(self, data: MetaTensor) -> MetaTensor:
        origin = data.affine[:self.spatial_dims, -1]
        spacing = data.pixdim
        spacing_sign = torch.sign(torch.diag(data.affine))[:self.spatial_dims]
        origin = origin * spacing_sign
        spatial_shape = torch.as_tensor(data.shape[-self.spatial_dims:])
        spatial_size = spatial_shape * spacing

        positions: list[torch.Tensor] = []
        for dim_index in range(self.spatial_dims):
            dim = spatial_shape[dim_index]
            # noinspection PyTypeChecker
            position = torch.arange(start=0, end=dim, step=1) / dim
            position = origin[dim_index] + position * spatial_size[dim_index]
            position = torch.sin(position * torch.pi * 0.5 / self.scale)
            position_shape = [dim if i == dim_index else 1 for i in range(self.spatial_dims)]
            position = position.view(position_shape)
            tile_shape = [1 if i == dim_index else spatial_shape[i] for i in range(self.spatial_dims)]
            position = position.tile(tile_shape)
            positions.append(position)

        global_position_embedding = torch.stack(positions, dim=0)
        if self.device in ["cuda", "gpu"]:
            global_position_embedding = global_position_embedding.cuda()
        elif self.device == "cpu":
            global_position_embedding = global_position_embedding.cpu()
        else:
            raise RuntimeError(self.device)

        return MetaTensor(global_position_embedding, affine=data.affine, meta=data.meta)

    @classmethod
    def json_identifier(cls) -> str:
        return "compute_global_position_embedding"

    def to_json(self) -> TransformParameters:
        return {
            "scale": self.scale,
            "spatial_dims": self.spatial_dims
        }


# endregion

# region Combine images

class ConcatenateChannels(SerializableTransform):
    def __init__(self, channel_dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.channel_dim = channel_dim

    def __call__(self, *data: torch.Tensor | MetaTensor) -> torch.Tensor | MetaTensor:
        return torch.concat(data, dim=self.channel_dim)

    @classmethod
    def json_identifier(cls) -> str:
        return "concatenate_channels"

    def to_json(self) -> TransformParameters:
        return {"channel_dim": self.channel_dim}


class PixelwiseMultiplication(SerializableTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, *data: torch.Tensor | MetaTensor) -> torch.Tensor | MetaTensor:
        if len(data) < 2:
            raise ValueError("Data must contain at least two tensors, got {}".format(len(data)))

        result = data[0] * data[1]
        for i in range(2, len(data)):
            result *= data[i]

        return result

    @classmethod
    def json_identifier(cls) -> str:
        return "pixelwise_multiplication"

    # noinspection PyMethodMayBeStatic
    def to_json(self) -> TransformParameters:
        return {}


# endregion

# region Dimension reduction

class TakeSlice(SerializableTransform):
    def __init__(self,
                 axis: int,
                 slice_index: int,
                 **kwargs):
        super().__init__(**kwargs)

        self.axis = axis
        self.slice_index = slice_index

    def __call__(self, data: torch.Tensor | MetaTensor) -> torch.Tensor | MetaTensor:
        indices = [self.slice_index if (i == self.axis) else slice(None) for i in range(len(data.shape))]
        return data[indices]

    @classmethod
    def json_identifier(cls) -> str:
        return "take_slice"

    def to_json(self) -> TransformParameters:
        return {
            "axis": self.axis,
            "slice_index": self.slice_index,
        }


class MaximumIntensityProjection(SerializableTransform):
    def __init__(self,
                 axis: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def __call__(self, data: np.ndarray | torch.Tensor | MetaTensor) -> np.ndarray | torch.Tensor | MetaTensor:
        if isinstance(data, np.ndarray):
            mip = data.max(axis=self.axis)
        else:
            mip, _ = data.max(dim=self.axis)
        return mip

    @classmethod
    def json_identifier(cls) -> str:
        return "maximum_intensity_projection"

    def to_json(self) -> TransformParameters:
        return {
            "axis": self.axis,
        }


# region 3D to 2D
class CenterSlicer3D(SerializableTransform):
    def __init__(self,
                 dims=(0, 1, 2),
                 dim_names=None,
                 **kwargs):
        super().__init__(**kwargs)
        if dim_names is None:
            dim_names = ["slice_{}".format(dim) for dim in dims]
        self.dims = dims
        self.dim_names = dim_names

    def __call__(self, image: torch.Tensor | MetaTensor) -> dict[str, torch.Tensor | MetaTensor]:
        ndim = len(image.shape)

        outputs = {}
        for dim, dim_name in zip(self.dims, self.dim_names):
            spatial_dim = dim + ndim - 3
            indices = [image.shape[i] // 2 if (i == spatial_dim) else slice(None) for i in range(ndim)]
            outputs[dim_name] = image[indices]

        return outputs

    @classmethod
    def json_identifier(cls) -> str:
        return "center_slicer_3d"

    def to_json(self) -> TransformParameters:
        return {
            "dims": self.dims,
            "dim_names": self.dim_names,
        }


class PriorSlicer(SerializableTransform):
    def __init__(self,
                 prior_loc: Sequence[int | float],
                 prior_scale: Sequence[int | float],
                 dims=(0, 1, 2),
                 dim_names=None,
                 **kwargs):
        super().__init__(**kwargs)
        if dim_names is None:
            dim_names = ["slice_{}".format(dim) for dim in dims]

        # region Check config
        dim_count = len(dims)
        misconfiguration = ((len(prior_loc) != dim_count)
                            or (len(prior_scale) != dim_count)
                            or (len(dim_names) != dim_count))
        if misconfiguration:
            raise ValueError("Misconfiguration: Expected to have {} dim values, got {}, {} and {}".
                             format(dim_count, len(prior_loc), len(prior_scale), len(dim_names)))
        # endregion

        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.dims = dims

        self.dim_names = dim_names

    def __call__(self, image: torch.Tensor | MetaTensor) -> dict[str, torch.Tensor | MetaTensor]:
        outputs = {dim_name: self.slice_with_prior(image, dim, prior_loc, prior_scale)
                   for dim, dim_name, prior_loc, prior_scale
                   in zip(self.dims, self.dim_names, self.prior_loc, self.prior_scale)}

        return outputs

    @staticmethod
    def slice_with_prior(image: torch.Tensor | MetaTensor,
                         dim: int,
                         prior_loc: int | float,
                         prior_scale: int | float
                         ) -> torch.Tensor | MetaTensor:
        ndim = len(image.shape)
        spatial_dim = dim + ndim - 3

        reduce_dims = [i for i in range(ndim) if i != spatial_dim]
        dim_mean_squared = image.pow(2.0).mean(reduce_dims)

        dim_size = image.shape[spatial_dim]
        dim_prior = normal_pdf(torch.arange(dim_size), loc=prior_loc, scale=prior_scale)

        slice_index = (dim_mean_squared * dim_prior).argmax()
        indices = [slice_index if i == spatial_dim else slice(None) for i in range(ndim)]

        return image[indices]

    @classmethod
    def json_identifier(cls) -> str:
        return "prior_slicer"

    def to_json(self) -> TransformParameters:
        return {
            "prior_loc": self.prior_loc,
            "prior_scale": self.prior_scale,
            "dims": self.dims,
            "dim_names": self.dim_names,
        }

# endregion

# endregion
