import torch
from monai.transforms import (
    Transform,
    LoadImage, EnsureChannelFirst, Orientation, Flip,
    CropForeground, SpatialCrop, CenterSpatialCrop, SpatialPad, ResizeWithPadOrCrop,
    Resize, Spacing, FgBgToIndices,
    ScaleIntensity, ScaleIntensityRange,
    RandGaussianNoise, RandFlip, RandRotate, RandRotate90, RandAffine, RandSpatialCrop, RandCropByPosNegLabel,
    RandShiftIntensity,
    EnsureType, ToTensor, ToDevice, Identity
)
from monai.transforms import (
    MapTransform,
    LoadImaged, EnsureChannelFirstd, Orientationd, Flipd,
    CropForegroundd, SpatialCropd, CenterSpatialCropd, SpatialPadd, ResizeWithPadOrCropd,
    Resized, Spacingd, FgBgToIndicesd,
    ScaleIntensityd, ScaleIntensityRanged,
    RandGaussianNoised, RandFlipd, RandRotated, RandRotate90d, RandAffined, RandSpatialCropd, RandCropByPosNegLabeld,
    RandShiftIntensityd,
    EnsureTyped, ToTensord, ToDeviced, Identityd
)
from monai.data import MetaTensor
import copy
from abc import ABC, abstractmethod
from typing import Any, Union, Optional

from data import Sample
from utils.misc import find_all_subclasses

TransformParameters = dict[str, Any]


class SerializableTransform(Transform, ABC):
    def __init__(self, **kwargs):
        pass

    # region Serialization
    # region Abstract methods
    @classmethod
    @abstractmethod
    def json_identifier(cls) -> str:
        raise NotImplementedError("This method must be implemented in subclasses.")

    @abstractmethod
    def to_json(self) -> TransformParameters:
        raise NotImplementedError("This method must be implemented in subclasses.")

    # endregion

    @staticmethod
    def serialize(transform: Transform) -> TransformParameters:
        if transform is None:
            return {}

        if isinstance(transform, SerializableTransform):
            return transform.to_json()

        if type(transform) not in supported_monai_transforms.values():
            raise NotImplementedError("type `{}` is not supported yet.".format(type(transform)))

        if isinstance(transform, MapTransform) and not isinstance(transform, RandRotate90d):
            return SerializableTransform.serialized(transform)

        # region Preprocessing
        # region Scan
        if isinstance(transform, LoadImage):
            config = {
                "image_only": transform.image_only,
                "dtype": str(transform.dtype),
                "ensure_channel_first": transform.ensure_channel_first,
                "simple_keys": transform.simple_keys,
                "prune_meta_pattern": transform.pattern,
                "prune_meta_sep": transform.sep,
            }
        elif isinstance(transform, EnsureChannelFirst):
            config = {
                "strict_check": transform.strict_check,
                "channel_dim": transform.input_channel_dim
            }
        elif isinstance(transform, Orientation):
            config = {
                "axcodes": transform.axcodes,
                "as_closest_canonical": transform.as_closest_canonical,
                "labels": transform.labels
            }
        elif isinstance(transform, ScaleIntensity):
            config = {
                "minv": transform.minv,
                "maxv": transform.maxv,
                "factor": transform.factor,
                "channel_wise": transform.channel_wise,
                "dtype": str(transform.dtype),
            }
        elif isinstance(transform, ScaleIntensityRange):
            config = {
                "a_min": transform.a_min,
                "a_max": transform.a_max,
                "b_min": transform.b_min,
                "b_max": transform.b_max,
                "clip": transform.clip,
                "dtype": str(transform.dtype),
            }
        elif isinstance(transform, CropForeground):
            config = {
                "channel_indices": transform.channel_indices,
                "margin": transform.margin,
                "allow_smaller": transform.allow_smaller,
                "return_coords": transform.return_coords,
                "k_divisible": transform.k_divisible,
                "mode": transform.padder.mode,
            }
        elif isinstance(transform, SpatialCrop):
            config = {
                "slices": str(transform.slices),
            }
        elif isinstance(transform, CenterSpatialCrop):
            config = {
                "roi_size": list(transform.roi_size)
            }
        elif isinstance(transform, SpatialPad):
            config = {
                "spatial_size": list(transform.spatial_size),
                "method": str(transform.method),
                "mode": str(transform.mode),
                "lazy": transform.lazy
            }
        elif isinstance(transform, ResizeWithPadOrCrop):
            config = {
                "spatial_size": list(transform.padder.spatial_size),
                "method": str(transform.padder.method),
                "mode": str(transform.padder.mode),
                "lazy": transform.lazy
            }
        elif isinstance(transform, Spacing):
            config = {
                "pixdim": transform.pixdim.tolist(),
                "min_pixdim": transform.min_pixdim.tolist(),
                "max_pixdim": transform.max_pixdim.tolist(),
                "diagonal": transform.diagonal,
                "mode": transform.sp_resample.mode,
                "padding_mode": transform.sp_resample.padding_mode,
                "align_corners": transform.sp_resample.align_corners,
                "dtype": str(transform.sp_resample.dtype),
                "scale_extent": transform.scale_extent,
                "recompute_affine": transform.recompute_affine,
                "lazy": transform.lazy,
            }
        elif isinstance(transform, FgBgToIndices):
            config = {
                "image_threshold": transform.image_threshold,
                "output_shape": transform.output_shape,
            }
        elif isinstance(transform, Resize):
            config = {
                "spatial_size": transform.spatial_size,
                "mode": transform.mode,
                "align_corners": transform.align_corners,
                "anti_aliasing": transform.anti_aliasing,
                "anti_aliasing_sigma": transform.anti_aliasing_sigma,
            }
        elif isinstance(transform, Flip):
            config = {
                "spatial_axis": transform.spatial_axis
            }
        # endregion

        # endregion
        # region Augmentations
        elif isinstance(transform, RandGaussianNoise):
            config = {
                "prob": transform.prob,
                "mean": transform.mean,
                "std": transform.std,
                "dtype": str(transform.dtype),
            }
        elif isinstance(transform, RandFlip):
            config = {
                "prob": transform.prob,
                "spatial_axis": transform.flipper.spatial_axis
            }
        elif isinstance(transform, RandRotate):
            config = {
                "prob": transform.prob,
                "range_x": transform.range_x,
                "range_y": transform.range_y,
                "range_z": transform.range_z,
                "padding_mode": transform.padding_mode,
            }
        elif isinstance(transform, (RandRotate90, RandRotate90d)):
            config = {
                "prob": transform.prob,
                "max_k": transform.max_k,
                "spatial_axes": transform.spatial_axes
            }
        elif isinstance(transform, RandAffine):
            config = {
                "prob": transform.prob,
                "translate_range": transform.rand_affine_grid.translate_range,
                "padding_mode": transform.padding_mode,
            }
        elif isinstance(transform, RandSpatialCrop):
            config = {
                "roi_size": transform.roi_size,
                "max_roi_size": transform.max_roi_size,
                "random_center": transform.random_center,
                "random_size": transform.random_size,
                "lazy": transform.lazy,
            }
        elif isinstance(transform, RandCropByPosNegLabel):
            config = {
                "spatial_size": transform.spatial_size,
                "label": None if transform.label is None else float(transform.label),
                "pos": 1.0,
                "neg": (1.0 / transform.pos_ratio) - 1.0,
                "num_samples": transform.num_samples,
                "image": None,
                "image_threshold": transform.image_threshold,
                "fg_indices": transform.fg_indices,
                "bg_indices": transform.bg_indices,
                "allow_smaller": transform.allow_smaller,
                "lazy": transform.lazy,
            }
        elif isinstance(transform, RandShiftIntensity):
            config = {
                "offsets": transform.offsets,
                "prob": transform.prob,
                "channel_wise": transform.channel_wise
            }
        # endregion
        elif isinstance(transform, EnsureType):
            config = {}
        elif isinstance(transform, ToDevice):
            config = {
                "device": str(transform.device)
            }
        elif isinstance(transform, ToTensor):
            config = {
                "dtype": str(transform.dtype),
                "device": str(transform.device),
                "wrap_sequence": transform.wrap_sequence
            }
        elif isinstance(transform, Identity):
            config = {}
        else:
            raise NotImplementedError(type(transform))

        return config

    @staticmethod
    def serialized(transform: MapTransform) -> TransformParameters:
        if isinstance(transform, LoadImaged):
            # noinspection PyProtectedMember
            base_transform = transform._loader
        elif isinstance(transform, Orientationd):
            base_transform = transform.ornt_transform
        elif isinstance(transform, ScaleIntensityd):
            base_transform = transform.scaler
        elif isinstance(transform, Spacingd):
            base_transform = transform.spacing_transform
        elif isinstance(transform, (CropForegroundd, SpatialCropd, CenterSpatialCropd, RandCropByPosNegLabeld)):
            base_transform = transform.cropper
        elif isinstance(transform, SpatialPadd):
            base_transform = transform.padder
        elif isinstance(transform, Resized):
            base_transform = transform.resizer
        elif isinstance(transform, ResizeWithPadOrCropd):
            base_transform = transform.padder
        elif isinstance(transform, (FgBgToIndicesd, ToDeviced, EnsureTyped)):
            base_transform = transform.converter
        elif isinstance(transform, RandRotated):
            base_transform = transform.rand_rotate
        elif isinstance(transform, RandAffined):
            base_transform = transform.rand_affine
        elif isinstance(transform, RandFlipd):
            base_transform = transform.flipper
        elif isinstance(transform, RandGaussianNoised):
            base_transform = transform.rand_gaussian_noise
        elif isinstance(transform, RandShiftIntensityd):
            base_transform = transform.shifter
        elif isinstance(transform, Identityd):
            base_transform = transform.identity
        elif isinstance(transform, EnsureChannelFirstd):
            base_transform = transform.adjuster
        else:
            raise NotImplementedError("Unimplemented MapTransform with type `{}`".format(type(transform)))

        return {
            "keys": transform.keys,
            **SerializableTransform.serialize(base_transform)
        }

    # endregion

    # region Deserialization
    @staticmethod
    def deserialize(identifier: str, parameters: TransformParameters, modalities: list[str] | None,
                    ) -> Union[Transform, "SerializableTransform"]:
        if identifier in supported_monai_transforms:
            if "type" in parameters:
                parameters = copy.copy(parameters)
                parameters.pop("type")
            transform_class = supported_monai_transforms[identifier]
            if issubclass(transform_class, MapTransform) and ("keys" not in parameters):
                parameters["keys"] = modalities
            return transform_class(**parameters)

        available_custom_transforms: set[type[SerializableTransform]] = find_all_subclasses(SerializableTransform)
        for available_custom_transform in available_custom_transforms:
            try:
                if identifier == available_custom_transform.json_identifier():
                    return available_custom_transform(**parameters)
            except NotImplementedError:
                pass

        raise NotImplementedError("Identifier `{}` does not match any currently supported transform.".
                                  format(identifier))

    @classmethod
    def default_outputs(cls) -> list[str] | None:
        return None

    @staticmethod
    def get_default_outputs(transform: Union[Transform, type]) -> Optional[list[str]]:
        if (isinstance(transform, SerializableTransform) or
                isinstance(transform, type) and issubclass(transform, SerializableTransform)):
            return transform.default_outputs()
        else:
            return None

    # endregion

    # region Fitting
    # noinspection PyUnusedLocal
    def fit(self, samples: list[Sample]) -> None:
        if self.requires_fitting():
            raise NotImplementedError("This method must be implemented in subclasses.")
        else:
            raise RuntimeError("This class does not support fitting on data.")

    def reset_preprocessed(self) -> None:
        if self.requires_fitting_preprocessed:
            raise NotImplementedError("This method must be implemented in subclasses.")
        else:
            raise RuntimeError("This class does not support fitting on preprocessed data.")

    # noinspection PyUnusedLocal
    def fit_preprocessed(self, sample: Any) -> None:
        if self.requires_fitting_preprocessed:
            raise NotImplementedError("This method must be implemented in subclasses.")
        else:
            raise RuntimeError("This class does not support fitting on preprocessed data.")

    def aggregate_preprocessed(self) -> None:
        if self.requires_fitting_preprocessed:
            raise NotImplementedError("This method must be implemented in subclasses.")
        else:
            raise RuntimeError("This class does not support fitting on preprocessed data.")

    # region Fitting - class methods
    # noinspection PyMethodMayBeStatic
    def requires_fitting(self) -> bool:
        return False

    @property
    def requires_fitting_preprocessed(self) -> bool:
        return False

    # endregion

    # region Fitting - static helpers
    @staticmethod
    def transform_requires_fitting(transform: Transform) -> bool:
        return isinstance(transform, SerializableTransform) and transform.requires_fitting()

    @staticmethod
    def transform_requires_fitting_preprocessed(transform: Transform) -> bool:
        return isinstance(transform, SerializableTransform) and transform.requires_fitting_preprocessed

    @staticmethod
    def get_transforms_to_fit(transforms: list[Transform]) -> list["SerializableTransform"]:
        # noinspection PyTypeChecker
        return [transform for transform in transforms
                if SerializableTransform.transform_requires_fitting(transform)]

    @staticmethod
    def get_transforms_to_fit_preprocessed(transforms: list[Transform]) -> list["SerializableTransform"]:
        # noinspection PyTypeChecker
        return [transform for transform in transforms
                if SerializableTransform.transform_requires_fitting_preprocessed(transform)]

    # endregion
    # endregion

    def add_metadata(self, tensor: torch.Tensor, meta_data: dict) -> MetaTensor:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        if not isinstance(tensor, MetaTensor):
            tensor = MetaTensor(tensor)

        if "transforms" not in tensor.meta:
            tensor.meta["transforms"] = {}

        tensor.meta["transforms"][self] = meta_data
        return tensor


supported_monai_transforms: dict[str, type[Transform] | type[MapTransform]] = {
    "load_image": LoadImage,
    "ensure_channel_first": EnsureChannelFirst,
    "orientation": Orientation,
    "scale_intensity": ScaleIntensity,
    "scale_intensity_range": ScaleIntensityRange,
    "spatial_crop": SpatialCrop,
    "center_spatial_crop": CenterSpatialCrop,
    "spatial_pad": SpatialPad,
    "resize_with_pad_or_crop": ResizeWithPadOrCrop,
    "spacing": Spacing,
    "fg_bg_to_indices": FgBgToIndices,
    "resize": Resize,
    "flip": Flip,
    "crop_foreground": CropForeground,
    "ensure_type": EnsureType,
    "rand_gaussian_noise": RandGaussianNoise,
    "rand_flip": RandFlip,
    "rand_rotate": RandRotate,
    "rand_rotate90": RandRotate90,
    "rand_affine": RandAffine,
    "rand_spatial_crop": RandSpatialCrop,
    "rand_crop_by_pos_neg_label": RandCropByPosNegLabel,
    "rand_shift_intensity": RandShiftIntensity,
    "to_tensor": ToTensor,
    "to_device": ToDevice,
    "identity": Identity,

    "load_imaged": LoadImaged,
    "ensure_channel_firstd": EnsureChannelFirstd,
    "orientationd": Orientationd,
    "scale_intensityd": ScaleIntensityd,
    "scale_intensity_ranged": ScaleIntensityRanged,
    "spatial_cropd": SpatialCropd,
    "center_spatial_cropd": CenterSpatialCropd,
    "spatial_padd": SpatialPadd,
    "resize_with_pad_or_cropd": ResizeWithPadOrCropd,
    "spacingd": Spacingd,
    "fg_bg_to_indicesd": FgBgToIndicesd,
    "resized": Resized,
    "flipd": Flipd,
    "crop_foregroundd": CropForegroundd,
    "ensure_typed": EnsureTyped,
    "rand_gaussian_noised": RandGaussianNoised,
    "rand_flipd": RandFlipd,
    "rand_rotated": RandRotated,
    "rand_rotate90d": RandRotate90d,
    "rand_affined": RandAffined,
    "rand_spatial_cropd": RandSpatialCropd,
    "rand_crop_by_pos_neg_labeld": RandCropByPosNegLabeld,
    "rand_shift_intensityd": RandShiftIntensityd,
    "to_tensord": ToTensord,
    "to_deviced": ToDeviced,
    "identityd": Identityd,
}

supported_monai_transforms_inv = {value: key for key, value in supported_monai_transforms.items()}
