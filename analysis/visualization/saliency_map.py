import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Union

from mindful_core.data import ModalitySet, Modality, ModalityType
from mindful_core.analysis.visualization.visualizer import Visualizer
from mindful_core.models.module import MindfulModule, ModelInput
from mindful_core.models.model_output import ClassifierOutput
from mindful_core.utils.tensor_utils import get_gradients, set_require_grads


class SaliencyMap(Visualizer):
    visualizer_name = "saliency"

    def __init__(self,
                 model: MindfulModule,
                 input_modalities: ModalitySet,
                 log_dir: str | Path,
                 smooth_maps: bool | int = True,
                 saved_versions: str | list[str] | tuple[str, ...] = ("center", "max_intensity", "depth_wise_max"),
                 color_map: str | int = cv2.COLORMAP_JET,
                 mask_threshold: float = 0.25,
                 overlay_coeff=0.1,
                 resize_method=cv2.INTER_LINEAR,
                 features_names: dict[Modality, list[str]] = None,
                 ):
        super(SaliencyMap, self).__init__(model,
                                          log_dir,
                                          saved_versions=saved_versions,
                                          color_map=color_map,
                                          mask_threshold=mask_threshold,
                                          overlay_coeff=overlay_coeff,
                                          resize_method=resize_method,
                                          input_modalities=input_modalities,
                                          features_names=features_names)
        self.smooth_maps = smooth_maps
        self._smoothing_kernel = None

    def __call__(self,
                 sample: list[torch.Tensor],
                 **kwargs
                 ) -> dict[str, torch.Tensor]:
        sample = [torch.unsqueeze(value, 0) for value in sample]
        was_training = self.model.training
        self.model.eval()
        saliency_maps, _ = self.get_saliency_maps(sample)
        self.model.train(was_training)

        if isinstance(saliency_maps, torch.Tensor):
            saliency_maps = [saliency_maps]

        result = {modality.id: torch.abs(modality_maps.detach())
                  for modality, modality_maps in zip(self.input_modalities, saliency_maps)}
        return result

    def run_batch(self, batch: list[torch.Tensor], ids: list[str] = None) -> None:
        inputs, labels, input_modalities = self.prepare_batch(batch)

        batch_size = self.get_batch_size(inputs)
        ids = self.get_ids(batch_size, ids)

        was_training = self.model.training
        self.model.eval()
        saliency_maps, predictions = self.get_saliency_maps(inputs)
        self.model.train(was_training)

        if isinstance(saliency_maps, torch.Tensor):
            saliency_maps = [saliency_maps]

        correct_predictions = self.get_correct_predictions(predictions, labels)
        kernel = get_3d_gaussian_kernel(size=3, sigma=1.0).to(predictions.logits.device)

        for modality, modality_value, modality_maps in zip(input_modalities, inputs, saliency_maps):
            if modality_maps is None:
                continue

            modality_value = modality_value.detach()
            modality_value = torch.abs(modality_value)

            saved_maps = modality_maps

            # Image (3D expected)
            if modality.type == ModalityType.IMAGE:
                modality_value = torch.movedim(modality_value, 1, -1).cpu().numpy()
                if self.use_smooth_maps:
                    smooth_modality_maps = blur_4d(modality_maps, kernel)
                    saved_maps = smooth_modality_maps

                    smooth_modality_maps = torch.movedim(smooth_modality_maps, 1, -1).cpu().numpy()
                    self.save_batch_output_slices(modality_value, smooth_modality_maps, ids,
                                                  correct_predictions, name="SmoothSaliency", modality=modality)

                modality_maps = torch.movedim(modality_maps, 1, -1).cpu().numpy()
                self.save_batch_output_slices(modality_value, modality_maps, ids,
                                              correct_predictions, name="Saliency", modality=modality)

            # Non-image
            elif modality.type in [ModalityType.SCALAR, ModalityType.CATEGORICAL]:
                modality_maps = modality_maps.cpu().numpy()
                self.save_features(modality, modality_maps, ids, "saliency")

            self.aggregate(modality, saved_maps, ids)

    def get_saliency_maps(self, inputs: list[torch.Tensor]) -> tuple[list[torch.Tensor], ClassifierOutput]:
        batch_size = inputs[0].size(0)
        sample_count = self.smooth_maps_sample_count

        gradients, predictions = [], []
        for i in range(batch_size):
            sample = [torch.unsqueeze(x[i], dim=0) for x in inputs]
            sample_gradients, sample_predictions = self.get_smooth_saliency_maps(sample, sample_count=sample_count)
            gradients.append(sample_gradients)
            predictions.append(sample_predictions)

        if isinstance(gradients[0], torch.Tensor):
            # noinspection PyTypeChecker
            gradients = torch.concat(gradients, dim=0)
        else:
            modality_count = len(gradients[0])
            gradients = [[gradients[i][j] for i in range(batch_size)] for j in range(modality_count)]
            gradients = [(torch.concat(x, dim=0) if x[0] is not None else None) for x in gradients]

        predictions = ClassifierOutput.concat(predictions)

        return gradients, predictions

    def get_smooth_saliency_maps(self,
                                 inputs: list[torch.Tensor],
                                 sample_count: int
                                 ) -> tuple[list[torch.Tensor], ClassifierOutput]:
        ###
        # This is a rather slow method because of the amount of backward passes
        ###
        if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
            inputs = inputs[0]

        gradients = []
        predictions: ClassifierOutput | None = None
        for _ in range(sample_count):
            noisy_inputs = self.to_noisy_inputs(inputs, stddev=0.01)

            noisy_inputs = set_require_grads(noisy_inputs)
            predictions = self.model(noisy_inputs)

            logits = predictions.logits

            if len(logits.shape) > 1:
                top_class = logits.argmax(dim=-1)
                top_predictions = logits[..., top_class]
            else:
                top_predictions = logits

            top_predictions.backward(torch.ones_like(top_predictions))
            sample_gradients = self.square_gradients(get_gradients(noisy_inputs))
            gradients.append(sample_gradients)

        if isinstance(gradients[0], torch.Tensor):
            gradients = torch.concat(gradients, dim=0)
        else:
            modality_count = len(gradients[0])
            gradients = [[gradients[i][j] for i in range(sample_count)] for j in range(modality_count)]
            gradients = [(torch.concat(x, dim=0) if x[0] is not None else None) for x in gradients]

        gradients = self.reduce_smooth_outputs(gradients)
        # predictions = self.reduce_smooth_outputs(predictions)

        return gradients, predictions
    
    @property
    def use_smooth_maps(self) -> bool:
        if isinstance(self.smooth_maps, bool):
            return self.smooth_maps
        else:
            return self.smooth_maps > 0
        
    @property
    def smooth_maps_sample_count(self) -> int:
        if isinstance(self.smooth_maps, bool):
            return 16 if self.smooth_maps else 1
        else:
            return self.smooth_maps

    @staticmethod
    def to_noisy_inputs(inputs: ModelInput,
                        stddev: float = 0.15
                        ) -> ModelInput:
        if isinstance(inputs, (list, tuple)):
            return [SaliencyMap.to_noisy_inputs(x, stddev) for x in inputs]

        if not torch.is_floating_point(inputs):
            return inputs

        noise = torch.randn_like(inputs) * stddev
        return inputs + noise

    @staticmethod
    def square_gradients(gradients: Union[torch.Tensor, list[torch.Tensor], None]
                         ) -> Union[torch.Tensor, list[torch.Tensor], None]:

        if isinstance(gradients, (list, tuple)):
            return [SaliencyMap.square_gradients(x) for x in gradients]

        if gradients is None:
            return None

        return torch.pow(gradients, 2.0)

    @staticmethod
    def reduce_smooth_outputs(outputs: Union[torch.Tensor, list[torch.Tensor], None],
                              ) -> Union[torch.Tensor, list[torch.Tensor], None]:
        if isinstance(outputs, (list, tuple)):
            return [SaliencyMap.reduce_smooth_outputs(x) for x in outputs]

        if outputs is None:
            return None

        return outputs.mean(dim=0, keepdim=True)


def index_to_gaussian_dist(index: int, size: int):
    coord = index - (size - 1) // 2
    return coord * coord


def get_3d_gaussian_kernel(size: int, sigma=1.0) -> torch.Tensor:
    kernel = np.zeros(shape=[size, size, size], dtype=np.float32)
    constant_factor = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    constant_power = - 1.0 / (2 * sigma * sigma)
    for i in range(size):
        x = index_to_gaussian_dist(i, size)
        for j in range(size):
            y = index_to_gaussian_dist(j, size)
            for k in range(size):
                z = index_to_gaussian_dist(k, size)
                kernel[i, j, k] = constant_factor * np.exp((x + y + z) * constant_power)

    kernel = torch.as_tensor(kernel, dtype=torch.float32)
    kernel = torch.reshape(kernel, [1, 1, size, size, size])
    return kernel


def blur_4d(image_4d: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    from torch.nn.functional import conv3d
    return conv3d(image_4d, kernel, padding="same")
