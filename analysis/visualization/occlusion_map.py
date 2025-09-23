import torch
from monai.transforms import Compose, GaussianSmooth, Lambda, ScaleIntensity
from monai.data.utils import dense_patch_slices
# noinspection PyProtectedMember
from monai.inferers.utils import _get_scan_interval
from monai.utils import ensure_tuple_rep
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Union, Sequence, Iterator

from data import ModalitySet, ModalityType, Modality
from analysis.visualization.visualizer import Visualizer
from models.model_output import ModelOutput, ClassifierOutput
from models.module import MindfulModule, ModelInput


class OcclusionMap(Visualizer):
    visualizer_name = "occlusion_map"

    def __init__(self,
                 model: MindfulModule,
                 input_modalities: ModalitySet,
                 log_dir: str | Path,
                 image_kernel_size: Union[int, list[int]],
                 image_kernel_overlap: float,
                 image_kernel_sigma=0.25,
                 saved_slices: str | list[str] | tuple[str, ...] = ("depth_wise_max",),
                 saved_versions: str | list[str] | tuple[str, ...] = ("next_to_overlay",
                                                                      "next_to_additive",
                                                                      "next_to_mul"),
                 color_map: str | int = cv2.COLORMAP_JET,
                 mask_threshold: float = 0.25,
                 overlay_coeff=0.1,
                 resize_method=cv2.INTER_LINEAR,
                 features_names: dict[Modality, list[str]] = None,
                 use_minimum: bool = False
                 ):
        super(OcclusionMap, self).__init__(model=model,
                                           log_dir=log_dir,
                                           saved_slices=saved_slices,
                                           saved_versions=saved_versions,
                                           color_map=color_map,
                                           mask_threshold=mask_threshold,
                                           overlay_coeff=overlay_coeff,
                                           resize_method=resize_method,
                                           input_modalities=input_modalities,
                                           features_names=features_names
                                           )
        self.image_kernel_size = image_kernel_size
        self.image_kernel_overlap = image_kernel_overlap
        self.image_kernel_sigma = image_kernel_sigma
        self.use_minimum = use_minimum
        self._occlusion_masks: dict[Modality, torch.Tensor] = {}

    def __call__(self,
                 sample: list[torch.Tensor],
                 prediction: ModelOutput = None,
                 **kwargs
                 ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            batched_sample = Visualizer.sample_as_batch(sample)
            batched_prediction: ModelOutput = self.model(batched_sample)
            prediction = batched_prediction[0]
            output = self.run_sample(sample, prediction)
            output = {modality.id: value for modality, value in output.items()}
        return output

    def run_batch(self, batch: list[torch.Tensor], ids: list[str] = None) -> None:
        inputs, labels, input_modalities = self.prepare_batch(batch)

        with torch.no_grad():
            predictions: ModelOutput = self.model(inputs)

            outputs = [self.run_sample(sample, sample_predictions)
                       for sample, sample_predictions
                       in tqdm(zip(self._iterate_batch(inputs), predictions),
                               total=len(inputs[0]))
                       ]
            outputs = self._stack_outputs(outputs)

        if isinstance(predictions, ClassifierOutput):
            correct_predictions = self.get_correct_predictions(predictions, labels)
        else:
            correct_predictions = None

        for modality_value, (modality, importance_map) in zip(inputs, outputs.items()):
            self.aggregate(modality, modality_value, ids)

            # Images (3D expected)
            if modality.type == ModalityType.IMAGE:
                self.save_images_3d(importance_map, ids, correct_predictions)

                modality_value = modality_value.squeeze(1).unsqueeze(-1).cpu().numpy()
                importance_map = importance_map.squeeze(1).unsqueeze(-1).cpu().numpy()
                self.save_batch_output_slices(modality_value, importance_map, ids,
                                              correct_predictions, name="OcclusionMap", modality=modality)

            # Non-image
            else:
                importance_map = importance_map.cpu().numpy()
                self.save_features(modality, importance_map, ids, name="occlusion")

    # region Batch I/O handling
    @staticmethod
    def _iterate_batch(batch: ModelInput) -> Iterator[ModelInput]:
        if isinstance(batch, torch.Tensor):
            for sample in batch:
                yield sample
        else:
            batch_size = batch[0].shape[0]
            for i in range(batch_size):
                yield [modality[i] for modality in batch]

    @staticmethod
    def _stack_outputs(outputs: list[dict[Modality, torch.Tensor]]) -> dict[Modality, torch.Tensor]:
        modalities = list(outputs[0].keys())
        stacked_outputs = {modality: torch.stack([output[modality] for output in outputs])
                           for modality in modalities}
        return stacked_outputs

    # endregion

    def run_sample(self, sample: ModelInput, base_predictions: ModelOutput) -> dict[Modality, torch.Tensor]:
        modalities = [sample] if isinstance(sample, torch.Tensor) else sample

        outputs = {modality: self._run_modality(modalities, i, modality.type, base_predictions)
                   for i, modality in enumerate(self.input_modalities)
                   if modality.type in (ModalityType.IMAGE, ModalityType.SCALAR, ModalityType.CATEGORICAL)}
        return outputs

    def _run_modality(self,
                      modalities: list[torch.Tensor],
                      modality_index: int,
                      modality_type: ModalityType,
                      base_predictions: ModelOutput
                      ) -> torch.Tensor:
        modality_value = modalities[modality_index]
        occlusion_masks = self.get_occlusion_masks(modality_value, modality_type)
        modalities = self.apply_occlusion_masks(modalities, modality_index, modality_type, occlusion_masks)

        batch_size = 512
        sample_count = len(modalities[0])
        predictions = []
        for i in range(0, sample_count, batch_size):
            batch = [modality[i:i + batch_size] for modality in modalities]
            if len(batch) == 1:
                batch = batch[0]
            predictions.append(self.model(batch))

        predictions = ClassifierOutput.concat(predictions)
        # predictions = self.model(modalities)

        distance = base_predictions.distance(predictions)
        importance_map = torch.einsum("b,b...->...", distance, 1 - occlusion_masks)
        return importance_map

    def get_occlusion_masks(self, tensor: torch.Tensor, modality_type: ModalityType) -> torch.Tensor:
        if modality_type == ModalityType.IMAGE:
            channel_count, *image_size = tensor.shape

            image_kernel_size = ensure_tuple_rep(self.image_kernel_size, dim=len(image_size))
            image_kernel_overlap = ensure_tuple_rep(self.image_kernel_overlap, dim=len(image_size))

            masks = self.build_gaussian_occlusion_masks(image_size=image_size,
                                                        kernel_size=image_kernel_size,
                                                        overlap=image_kernel_overlap,
                                                        channel_count=channel_count,
                                                        device=tensor.device,
                                                        dtype=tensor.dtype,
                                                        sigma=self.image_kernel_sigma)
        elif modality_type in (ModalityType.SCALAR, ModalityType.CATEGORICAL):
            feature_count = tensor.shape[-1]
            masks = self.build_feature_wise_occlusion_masks(feature_count,
                                                            device=tensor.device,
                                                            dtype=tensor.dtype)

        else:
            raise RuntimeError(modality_type)

        return masks

    def apply_occlusion_masks(self,
                              modalities: list[torch.Tensor],
                              modality_index: int,
                              modality_type: ModalityType,
                              occlusion_masks: torch.Tensor
                              ) -> list[torch.Tensor]:
        masks_count = occlusion_masks.shape[0]
        modalities = [(self.apply_occlusion_mask(modality, occlusion_masks, modality_type)
                       if (i == modality_index)
                       else self.expand_modality(modality, masks_count))
                      for i, modality in enumerate(modalities)]
        return modalities

    def apply_occlusion_mask(self,
                             modality: torch.Tensor,
                             occlusion_masks: torch.Tensor,
                             modality_type: ModalityType
                             ) -> torch.Tensor:
        modality = torch.unsqueeze(modality, 0)
        if (modality_type == ModalityType.IMAGE) and self.use_minimum:
            min_value = modality.min()
            return modality * occlusion_masks + (1 - occlusion_masks) * min_value
        else:
            return modality * occlusion_masks

    @staticmethod
    def expand_modality(modality_value: torch.Tensor, batch_size: int) -> torch.Tensor:
        expansion = [batch_size] + ([-1] * len(modality_value.shape))
        return modality_value.unsqueeze(0).expand(*expansion)

    # region N-D masking (N > 0)
    @staticmethod
    def build_gaussian_kernel(kernel_size: Sequence[int],
                              channel_count: int = 1,
                              device: torch.device = None,
                              dtype: torch.dtype = None,
                              sigma: float = 0.25):
        """
        Based on monai.visualize.occlusion_sensitivity.OcclusionSensitivity
        """
        kernel = torch.zeros(channel_count, *kernel_size, device=device, dtype=dtype)
        center = [slice(None)] + [slice(s // 2, s // 2 + 1) for s in kernel_size]
        kernel[center] = 1.0

        gaussian = Compose([
            GaussianSmooth(sigma=[b * sigma for b in kernel_size]),
            Lambda(lambda x: -x),
            ScaleIntensity()
        ])

        kernel = gaussian(kernel)

        return kernel

    @staticmethod
    def build_gaussian_occlusion_masks(image_size: Sequence[int],
                                       kernel_size: Sequence[int],
                                       overlap: Sequence[float],
                                       channel_count: int = 1,
                                       device: torch.device = None,
                                       dtype: torch.dtype = None,
                                       sigma: float = 0.25
                                       ) -> torch.Tensor:
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(image_size)
        elif len(image_size) != len(kernel_size):
            raise ValueError("Expected `image_size` and `kernel_size` to have the same length (got {} and {})".
                             format(len(image_size), len(kernel_size)))

        image_rank = len(image_size)

        scan_interval = _get_scan_interval(image_size, kernel_size, image_rank, overlap)
        slices = dense_patch_slices(image_size, kernel_size, scan_interval)
        slices = [[slice(0, channel_count), *_slice] for _slice in slices]

        kernel = OcclusionMap.build_gaussian_kernel(kernel_size, channel_count, device, dtype, sigma)

        mask_shape = (channel_count, *image_size)
        masks = []
        for _slice in slices:
            mask = torch.ones(mask_shape, device=device, dtype=dtype)
            mask[_slice] = kernel
            masks.append(mask)

        return torch.stack(masks, dim=0)

    # endregion

    # region 0-D masking

    @staticmethod
    def build_feature_wise_occlusion_masks(feature_count: int,
                                           device: torch.device = None,
                                           dtype: torch.dtype = None,
                                           ) -> torch.Tensor:
        return 1 - torch.eye(feature_count, device=device, dtype=dtype)

    # endregion
