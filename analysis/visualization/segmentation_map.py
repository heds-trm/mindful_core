import torch
from monai.inferers import SlidingWindowInferer, SimpleInferer
import cv2
from pathlib import Path
from typing import Sequence

from data import ModalitySet
from analysis.visualization.visualizer import Visualizer
from models.segmentation.segmentation_unet import SegmentationUNet, SegmentationOutput
from utils.imaging import apply_palette


class SegmentationMap(Visualizer):
    visualizer_name = "segmentation_maps"

    def __init__(self,
                 model: SegmentationUNet,
                 log_dir: str | Path,
                 roi_size: Sequence[int] | None = None,
                 sw_batch_size: int = 128,
                 saved_slices: str | list[str] | tuple[str, ...] = ("center",),
                 saved_versions: str | list[str] | tuple[str, ...] = ("next_to_segmentation",),
                 color_map: str | int = None,
                 mask_threshold: float = 0.25,
                 overlay_coeff=0.1,
                 resize_method=cv2.INTER_LINEAR,
                 input_modalities: ModalitySet = None,
                 **kwargs
                 ):
        super().__init__(model=model,
                         log_dir=log_dir,
                         saved_slices=saved_slices,
                         saved_versions=saved_versions,
                         color_map=color_map,
                         mask_threshold=mask_threshold,
                         overlay_coeff=overlay_coeff,
                         resize_method=resize_method,
                         input_modalities=input_modalities
                         )

        self.model: SegmentationUNet
        self.roi_size = roi_size
        if roi_size is None:
            self.inferer = SimpleInferer()
        else:
            self.inferer = SlidingWindowInferer(roi_size, sw_batch_size)

        self._kwargs = kwargs

    def __call__(self,
                 sample: list[torch.Tensor] | torch.Tensor,
                 **kwargs
                 ) -> dict[str, torch.Tensor]:
        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            sample = Visualizer.sample_as_batch(sample)
            if not isinstance(sample, torch.Tensor):
                if len(sample) != 2:
                    raise NotImplementedError("Expected at most 2 modalities (image and mask)")
                image, mask = sample
            else:
                image, mask = sample, None

            batched_logits = self.inferer(image, self.model.backbone)
            prediction = batched_logits[0].argmax(dim=0, keepdim=True)

        return prediction

    def run_batch(self, batch: list[torch.Tensor] | torch.Tensor, ids: list[str] = None) -> None:
        if not isinstance(batch, torch.Tensor):
            if len(batch) != 2:
                raise NotImplementedError("Expected at most 2 modalities (image and mask)")
            images, masks = batch
        else:
            images, masks = batch, None

        with torch.no_grad():
            outputs: SegmentationOutput = self.model(images)
            segmentations = outputs.logits.argmax(dim=1, keepdim=True)

        self.save_images_3d(segmentations, ids)

        segmentations = apply_palette(segmentations)
        masks = apply_palette(masks)

        images = images.permute(0, 2, 3, 4, 1)
        segmentations = segmentations.permute(0, 2, 3, 4, 1)
        masks = masks.permute(0, 2, 3, 4, 1)

        self.save_batch_output_slices(images, segmentations, ids, None, "SegmentationMap",
                                      modality="segmentation")
        if masks is not None:
            self.save_batch_output_slices(images, masks, ids, None, "SegmentationMasks",
                                          modality="mask")
