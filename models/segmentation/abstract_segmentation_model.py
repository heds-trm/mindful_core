import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.segmentation import generalized_dice_score
from pytorch_lightning.utilities.types import STEP_OUTPUT
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.data.utils import dense_patch_slices
from abc import abstractmethod
from typing import Type, Any, Sequence

from utils.tensor_utils import normalize
from utils.imaging import grayscale_to_rgb, index_to_tricolor, apply_palette
from data.subset_id import SubsetID
from models.model_output import SegmentationOutput
from models.module import MindfulModule


class AbstractSegmentationModel(MindfulModule):
    # region Class/Subclass methods
    @classmethod
    @abstractmethod
    def module_identifier(cls) -> str:
        raise NotImplementedError("Method `module_identifier` must be implemented in subclasses")

    @classmethod
    def get_registered_segmentation_models(cls) -> dict[str, Type["AbstractSegmentationModel"]]:
        return {key: value for key, value in cls._module_register if issubclass(value, AbstractSegmentationModel)}

    @classmethod
    def base_log_monitors(cls) -> list[str]:
        return ["loss", "dice"]

    # endregion

    def __init__(self,
                 class_count: int,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 roi_size: Sequence[int] | None = None,
                 ):
        super().__init__(optimizer_config=optimizer_config)

        self.class_count = class_count
        self.roi_size = roi_size

    @abstractmethod
    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> SegmentationOutput:
        raise NotImplementedError("Method `forward` must be implemented in subclasses")

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        return self.base_step(batch, subset_id=SubsetID.TRAIN)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        with torch.no_grad():
            return self.base_step(batch, subset_id=SubsetID.VALIDATION)

    def test_step(self,
                  batch: tuple[torch.Tensor, torch.Tensor],
                  *args,
                  **kwargs) -> STEP_OUTPUT | None:
        with torch.no_grad():
            inputs, _ = batch
            if self.sliding_window_required(inputs):
                return self.sliding_base_step(batch, subset_id=SubsetID.TEST)
            else:
                return self.base_step(batch, subset_id=SubsetID.TEST)

    def base_step(self,
                  batch: tuple[torch.Tensor, torch.Tensor],
                  subset_id: SubsetID,
                  **model_kwargs) -> STEP_OUTPUT:
        self.loss_aggregator.clear()
        inputs, targets = batch
        targets = targets.to(torch.int64)

        outputs: SegmentationOutput = self(inputs, **model_kwargs)
        loss: torch.Tensor = self._loss_module(outputs.logits, targets)
        self.loss_aggregator.add_loss(loss.mean(), "base_loss")

        self.log_step_losses(subset_id)
        loss, logged_losses = self.loss_aggregator.get_logged_losses()

        logits = outputs.logits.detach()
        with torch.no_grad():
            metrics = self.compute_metrics(logits, targets)

        if len(self.step_outputs[subset_id]) == 0:
            predictions = logits.argmax(dim=1, keepdim=True)
            self.log_images(inputs, targets, predictions, subset_id)

        metrics.update(logged_losses)

        self.step_outputs[subset_id].append(metrics)
        return loss

    # region Sliding window
    def sliding_base_step(self,
                          batch: tuple[torch.Tensor, torch.Tensor],
                          subset_id: SubsetID,
                          **model_kwargs) -> STEP_OUTPUT:
        inputs, targets = batch

        inputs, targets = self.spatial_pad_to_max(inputs, targets)
        inputs = self.pad_and_batch_patches(inputs)
        targets = self.pad_and_batch_patches(targets)

        batch = (inputs, targets)
        return self.base_step(batch, subset_id, **model_kwargs)

    @staticmethod
    def spatial_pad_to_max(*tensors: torch.Tensor) -> list[torch.Tensor]:
        tensors = list(tensors)
        tensors_shapes = [tensor.shape for tensor in tensors]
        max_shape = [max(*dim_sizes) for dim_sizes in zip(*tensors_shapes)]

        for i in range(len(tensors)):
            pad_size = []
            pad_required = False
            for axis in range(len(max_shape) - 1, 1, -1):
                diff = max_shape[axis] - tensors_shapes[i][axis]
                half = diff // 2
                pad_size.extend([half, diff - half])
                if diff > 0:
                    pad_required = True

            if pad_required:
                tensors[i] = torch.nn.functional.pad(tensors[i], pad=pad_size, mode="constant", value=0.0)

        return tensors

    def pad_and_batch_patches(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, inputs_channels, *_ = tensor.shape

        tensor = self.pad_for_sliding_window(tensor)
        patch_size = [batch_size, inputs_channels, *self.roi_size]
        slices = dense_patch_slices(tensor.shape, patch_size, patch_size)
        tensor = self.batch_patches(tensor, slices)

        return tensor

    def sliding_window_required(self,
                                inputs: torch.Tensor,
                                ) -> bool:
        if self.roi_size is None:
            return False
        spatial_shape = inputs.shape[2:]
        return spatial_shape != self.roi_size

    def pad_for_sliding_window(self, tensor: torch.Tensor) -> torch.Tensor:
        pad_size = []
        pad_required = False
        for i in range(len(tensor.shape) - 1, 1, -1):
            step = self.roi_size[i - 2]
            size = tensor.shape[i]
            diff = step - (size % step)
            half = diff // 2
            pad_size.extend([half, diff - half])
            if diff > 0:
                pad_required = True

        if pad_required:
            tensor = torch.nn.functional.pad(tensor, pad=pad_size, mode="constant", value=0.0)

        return tensor

    @staticmethod
    def batch_patches(tensor: torch.Tensor, slices: list[tuple[slice, ...]]) -> torch.Tensor:
        return torch.concat([tensor[_slice] for _slice in slices], dim=0)

    # endregion

    # region Metrics
    @staticmethod
    def compute_metrics(logits: torch.Tensor,
                        targets: torch.Tensor,
                        predictions: torch.Tensor | None = None,
                        ) -> dict[str, torch.Tensor]:
        if predictions is None:
            predictions = logits.argmax(dim=1, keepdim=True)

        outputs: dict[str, torch.Tensor] = {}

        # region Binary DICEs
        binary_targets = (0 == targets).long()
        binary_targets = torch.concat([binary_targets, 1 - binary_targets], dim=1)
        # region Max-class DICE
        #   Refers to picking the non-background class with the max logits vs the background class
        max_class_predictions = (0 == predictions).long()
        max_class_predictions = torch.concat([max_class_predictions, 1 - max_class_predictions], dim=1)
        max_class_dice = generalized_dice_score(max_class_predictions, binary_targets,
                                                num_classes=2, include_background=False)
        outputs["max_class_dice"] = max_class_dice
        outputs["dice"] = outputs["max_class_dice"]
        # endregion

        # endregion

        return outputs

    def log_epoch_outputs(self, subset: SubsetID):
        if self.trainer.sanity_checking:
            return {}

        outputs = self.step_outputs[subset]
        suffix = "epoch"
        metrics = self.aggregate_epoch_metrics(outputs)
        self.log_epoch_metrics(metrics, subset, suffix=suffix)

    @staticmethod
    def aggregate_epoch_metrics(epoch_metrics: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not isinstance(epoch_metrics, list):
            raise ValueError("`outputs` must either be a list.")

        keys = []
        for step_output in epoch_metrics:
            for key in step_output:
                if key not in keys:
                    keys.append(key)

        metrics = {key: AbstractSegmentationModel.aggregate_batched_metrics([batch[key] for batch in epoch_metrics
                                                                             if key in batch])
                   for key in keys}

        return metrics

    @staticmethod
    def aggregate_batched_metrics(metrics: list[torch.Tensor]) -> torch.Tensor:
        if len(metrics[0].shape) == 0:
            return torch.stack(metrics).mean()
        else:
            return torch.concat(metrics).mean()

    # endregion

    # region Image logging
    def log_images(self,
                   inputs: torch.Tensor,
                   targets: torch.Tensor,
                   predictions: torch.Tensor,
                   subset_id: SubsetID):
        if self.logger is None:
            return

        prefix = subset_id.as_prefix()
        # noinspection PyUnresolvedReferences
        writer: SummaryWriter = self.logger.experiment
        inputs, targets, predictions = inputs[:1], targets[:1], predictions[:1]
        if inputs.shape[1] > 1:
            # remove extra channels if they exist
            inputs = inputs[:, :1]

        joint = self.join_segmentation_images(inputs, targets, predictions)

        plot_2d_or_3d_image(inputs, self.global_step, writer, tag="{}_input".format(prefix))
        plot_2d_or_3d_image(targets, self.global_step, writer, tag="{}_target".format(prefix))
        plot_2d_or_3d_image(predictions, self.global_step, writer, tag="{}_prediction".format(prefix))
        plot_2d_or_3d_image(joint, self.global_step, writer, tag="{}_joint".format(prefix), max_channels=3)

    def join_segmentation_images(self,
                                 inputs: torch.Tensor,
                                 targets: torch.Tensor,
                                 predictions: torch.Tensor) -> torch.Tensor:
        inputs = grayscale_to_rgb(inputs)
        if self.class_count <= 4:
            targets = index_to_tricolor(targets).to(inputs.dtype)
            predictions = index_to_tricolor(predictions).to(inputs.dtype)
        else:
            targets = apply_palette(targets).to(inputs.dtype)
            predictions = apply_palette(predictions).to(inputs.dtype)

        images = [inputs, targets, predictions]
        images = [normalize(image) for image in images]

        images = torch.concat(images, dim=-1)
        return images

    # endregion

    @property
    def single_class(self) -> bool:
        return self.class_count == 1
