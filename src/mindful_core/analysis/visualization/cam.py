import torch
from monai.visualize import CAM, GradCAM
import cv2
from pathlib import Path
from typing import Callable

from mindful_core.data import ModalitySet, ModalityType
from mindful_core.analysis.visualization.visualizer import Visualizer
from mindful_core.models.classification.monai_classifier import MonaiClassifier
from mindful_core.utils.tensor_utils import normalize


class CAMs(Visualizer):
    model: MonaiClassifier

    visualizer_name = "cams"

    def __init__(self,
                 model: MonaiClassifier,
                 log_dir: str | Path,
                 saved_slices: str | list[str] | tuple[str, ...] = "center",
                 saved_versions: str | list[str] | tuple[str, ...] = ("overlay", "next_to_overlay", "hsv"),
                 color_map: str | int = cv2.COLORMAP_JET,
                 mask_threshold: float = 0.25,
                 overlay_coeff=0.1,
                 resize_method=cv2.INTER_LINEAR,
                 input_modalities: ModalitySet = None
                 ):
        super(CAMs, self).__init__(model=model,
                                   log_dir=log_dir,
                                   saved_slices=saved_slices,
                                   saved_versions=saved_versions,
                                   color_map=color_map,
                                   mask_threshold=mask_threshold,
                                   overlay_coeff=overlay_coeff,
                                   resize_method=resize_method,
                                   input_modalities=input_modalities
                                   )

        if not isinstance(model, MonaiClassifier):
            raise NotImplementedError("This visualizer only works with MonaiClassifiers at the moment.")
        elif model.class_count == 1:
            raise NotImplementedError("CAM: Binary classifiers are currently unsupported.")

        self.cam = CAM(model, model.get_target_layer_name(), model.get_fully_connected_layer_name())
        self.grad_cam = GradCAM(model, model.get_target_layer_name())

        if not isinstance(saved_slices, (tuple, list)):
            saved_slices = (saved_slices,)
        self.saved_slices = saved_slices

    def __call__(self,
                 sample: list[torch.Tensor],
                 **kwargs
                 ) -> dict[str, torch.Tensor | None]:
        result = {}
        for modality, input_value in zip(self.input_modalities, sample):
            if modality.type != ModalityType.IMAGE:
                result[modality.id] = None
                continue

            # todo: NYI for multimodal setups
            cam_outputs = self.grad_cam(input_value)
            cam_outputs = torch.movedim(cam_outputs, 1, -1).cpu()
            result[modality.id] = cam_outputs

        return result

    def run_batch(self, batch: list[torch.Tensor], ids: list[str] = None) -> None:
        inputs, labels, _ = self.prepare_batch(batch)
        if labels is None:
            raise NotImplementedError("Only lists containing inputs and labels are supported at the moment.")

        batch_size = self.get_batch_size(inputs)
        ids = self.get_ids(batch_size, ids)

        if len(inputs) == 1:
            inputs = inputs[0]
        else:
            raise NotImplementedError("Multimodal support is not implemented yet.")

        self.model.disable_confidence = True

        # noinspection PyCallingNonCallable
        logits = self.model(inputs)
        if len(logits.shape) == 1 or logits.shape[-1] == 1:
            raise NotImplementedError("Not yet implemented for 1-class models.")
        predicted_class = torch.argmax(logits, dim=-1)
        correct_predictions: torch.Tensor = labels == predicted_class

        self._run_cam(self.cam, inputs, correct_predictions, cam_name="CAM", ids=ids, save_input=True)
        self._run_cam(self.grad_cam, inputs, correct_predictions, cam_name="GradCAM", ids=ids, save_input=False)

        self.model.disable_confidence = False

    def _run_cam(self,
                 cam: Callable[[torch.Tensor], torch.Tensor],
                 inputs: torch.Tensor,
                 correct_predictions: torch.Tensor,
                 cam_name: str,
                 save_input: bool,
                 ids: list[str]) -> None:
        if len(correct_predictions) != len(inputs):
            raise ValueError("The length of inputs and correct_predictions must match (got {} and {}).".
                             format(len(correct_predictions), len(inputs)))

        correct_predictions = correct_predictions.cpu().numpy()
        cam_outputs = cam(inputs)
        cam_outputs = torch.movedim(cam_outputs, 1, -1).cpu().numpy()
        inputs = torch.movedim(inputs, 1, -1).cpu().numpy()

        for cam_input, cam_output, sample_id, correct_prediction in zip(inputs, cam_outputs, ids, correct_predictions):
            self._save_output_slices(cam_output, cam_name, sample_id, correct_prediction, cam_input)
            if save_input:
                cam_input = normalize(cam_input)

                mid_index = (cam_input.shape[-2] - 1) // 2
                cam_input = cam_input[..., mid_index, :]

                cam_input = self._format_image(cam_input, normalize_image=False)

                self.write_image(cam_input, correct_prediction, sample_id, image_name="Input")
