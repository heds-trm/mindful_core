import torch
import torch.nn as nn
from torch.utils import data
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision.transforms.functional import to_pil_image
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
from abc import abstractmethod
from typing import Union, Sequence, Callable, Any

from mindful_core.models.model_output import RepresentationOutput
from mindful_core.models.module import MindfulModule
from mindful_core.data.subset_id import SubsetID
from mindful_core.models.representation.callbacks import RepresentationModelCallback, Projector, SeparabilityLogger
from mindful_core.models.loss_aggregator import LossAggregator
from mindful_core.utils.tensor_utils import to_numpy


def map_mean(function: Callable, sequence: Sequence) -> torch.Tensor | float:
    result = sum([function(x) for x in sequence]) / len(sequence)
    return result


class AbstractRepresentationModel(MindfulModule):
    @classmethod
    def base_log_monitors(cls) -> list[str]:
        return ["base_loss", "loss"]

    def __init__(self,
                 output_dimension: int,
                 covariance_lambda: float,
                 variance_lambda: float,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 callback_configs: dict[str, dict[str, Any]] = None,
                 *args, **kwargs):
        if "log_projector_period" in kwargs:
            log_projector_period = kwargs.pop("log_projector_period")
            callback_configs = {"projector": {"frequency": log_projector_period, "max_count": 1000}}

        super(AbstractRepresentationModel, self).__init__(optimizer_config=optimizer_config, *args, **kwargs)
        self.output_dimension = output_dimension
        self.variance_lambda = variance_lambda
        self.covariance_lambda = covariance_lambda
        self._representation_callbacks: list[RepresentationModelCallback] = []

        if callback_configs is not None:
            self.make_representation_callbacks(callback_configs)

    def forward(self,
                inputs: torch.Tensor,
                *args,
                flatten: bool = True,
                output_intermediates: bool = False,
                **kwargs
                ) -> RepresentationOutput:
        raise NotImplementedError("`forward` must be implemented in subclasses.")

    @staticmethod
    def flatten_representation(representations):
        if len(representations.shape) > 2:
            representations = torch.flatten(representations, start_dim=1)
        return representations

    @staticmethod
    def _get_model_representations(model: nn.Module,
                                   inputs: torch.Tensor,
                                   flatten: bool,
                                   output_intermediates: bool = False
                                   ) -> RepresentationOutput:
        outputs = model(inputs, output_intermediates=output_intermediates)

        if isinstance(outputs, (tuple, list)):
            representations, intermediates = outputs
        else:
            representations, intermediates = outputs, None

        if flatten:
            representations = representations.flatten(start_dim=1)

            if (intermediates is not None) and output_intermediates:
                intermediates = [intermediate.flatten(start_dim=1) for intermediate in intermediates]

        return RepresentationOutput(representations, intermediates)

    # region Training
    @abstractmethod
    def base_step(self, inputs, subset_id: SubsetID, *args, **kwargs) -> STEP_OUTPUT:
        raise NotImplementedError("`base_step` must be implemented in subclasses.")

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        return self.base_step(batch, subset_id=SubsetID.TRAIN)

    def validation_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        return self.base_step(batch, subset_id=SubsetID.VALIDATION)

    def compute_variance_covariance_loss(self,
                                         representations: Union[torch.Tensor, tuple[torch.Tensor, ...]]
                                         ) -> LossAggregator:
        if self.add_variance_covariance_loss:
            variance_loss = self.get_batch_variance_loss(representations)
            covariance_loss = self.get_covariance_loss(representations)

            self.loss_aggregator.add_loss(variance_loss, "variance_loss", self.variance_lambda)
            self.loss_aggregator.add_loss(covariance_loss, "covariance_loss", self.covariance_lambda)

        return self.loss_aggregator

    # region Variance / Covariance loss (cf. VICReg paper)
    @staticmethod
    def get_batch_variance_loss(representations: Union[torch.Tensor, Sequence[torch.Tensor]]
                                ) -> torch.Tensor:
        """
            Loss function aiming at maximizing the variance of each feature independently across the batch.
        """
        if not isinstance(representations, torch.Tensor):
            return map_mean(AbstractRepresentationModel.get_batch_variance_loss, representations)

        batch_std = torch.sqrt(representations.var(dim=0) + 1e-4)

        one = torch.ones(1, device=batch_std.device, dtype=batch_std.dtype)
        variance_loss = torch.mean(torch.relu(one - batch_std))

        return variance_loss

    @staticmethod
    def get_covariance_loss(representations: Union[torch.Tensor, Sequence[torch.Tensor]]
                            ) -> torch.Tensor:
        """
            Loss function aiming at minimizing the covariance between features (for each sample independently ?)
        """
        if not isinstance(representations, torch.Tensor):
            return map_mean(AbstractRepresentationModel.get_covariance_loss, representations)

        batch_size, latent_size = representations.shape

        representations_mean = representations.mean(dim=0, keepdim=True)
        representations = representations - representations_mean

        covariance = (representations.T @ representations) / (batch_size - 1)  # [latent_size, latent_size]
        # region Remove diagonal
        covariance = covariance.flatten()  # [latent_size * latent_size]
        covariance = covariance[:-1]  # [latent_size * latent_size - 1]
        covariance = covariance.view(latent_size - 1, latent_size + 1)  # [latent_size - 1, latent_size + 1]
        covariance = covariance[:, 1:]  # [latent_size - 1, latent_size]
        # endregion

        covariance_regularization = covariance.pow(2.0).sum() / latent_size
        return covariance_regularization

    # endregion

    # endregion

    # region Logs (metrics & representations)
    # region Init
    def make_representation_callbacks(self, configs: dict[str, dict[str, Any]]):
        for callback_class, callback_config in configs.items():
            callback = self.make_representation_callback(callback_class, callback_config)
            self.add_representation_callback(callback)

    @staticmethod
    def make_representation_callback(callback_class: str, config: dict[str, Any]) -> RepresentationModelCallback:
        callback_class = callback_class.lower()

        if "projector" in callback_class:
            return Projector(**config)

        if "separability" in callback_class:
            return SeparabilityLogger(**config)

        raise RuntimeError("Unknown representation callback class: {}.".format(callback_class))

    def add_representation_callback(self, representation_callback: RepresentationModelCallback):
        self._representation_callbacks.append(representation_callback)
        representation_callback.logger = self.logger
        representation_callback.logger_fn = self.get_logger

    def get_logger(self):
        return self.logger

    # endregion

    @staticmethod
    def _extract_representations(outputs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        outputs = [batch["representations"] for batch in outputs]
        outputs = torch.concat(outputs, dim=0)
        return outputs

    def log_epoch_outputs(self, subset: SubsetID):
        outputs = self.step_outputs[subset]

        metrics = {
            key: torch.mean(torch.stack([batch[key] for batch in outputs], dim=0))
            for key in outputs[0] if key not in ["representations"]
        }

        self.log_epoch_metrics(metrics, subset)
        self.on_representation_end(outputs, subset)

    def on_representation_end(self,
                              outputs: Union[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]],
                              subset_id: SubsetID):
        if not any([callback.uses_subset(subset_id) for callback in self._representation_callbacks]):
            return

        representations = self._extract_representations(outputs)

        for representation_callback in self._representation_callbacks:
            representation_callback.on_representation_end(representations=representations,
                                                          metadata=None,
                                                          metadata_header=None,
                                                          global_step=self.current_epoch,
                                                          subset=subset_id)

    # endregion

    # region K closest representations
    def get_k_closest(self,
                      data_loader: data.DataLoader,
                      k: int = 5
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # noinspection GrazieInspection
        """
        Compute the scores and indices of the k closest representations in the data_loader. Only suitable for relatively
        small datasets (as all representations are computed before they are compared).
        :param data_loader:
        :param k:
        :return: Similarity scores of the k closest representations, their indices and the inputs
        """
        with torch.no_grad():
            all_inputs = []
            representations = []

            for batch in data_loader:
                all_inputs.append(batch)
                representations.append(self(batch))

            all_inputs = torch.cat(all_inputs, dim=0)
            representations = torch.cat(representations, dim=0)

        if len(representations.shape) > 2:
            sample_count = representations.size(0)
            representations = representations.reshape(sample_count, -1)

        similarity_scores = torch.einsum("nd,nd->nn", representations, representations)
        top_scores, top_indices = torch.topk(similarity_scores, k=k, dim=-1)

        return top_scores, top_indices, all_inputs

    def plot_k_closest(self,
                       data_loader: data.DataLoader,
                       k: int = 5):
        top_scores, top_indices, all_inputs = self.get_k_closest(data_loader, k)

        top_scores = to_numpy(top_scores)
        top_indices = to_numpy(top_indices)
        all_inputs = to_numpy(all_inputs)

        figure = Figure(figsize=(12, 5))
        canvas = FigureCanvas()

        axes = figure.subplots(nrows=1, ncols=k)
        for i in range(len(top_scores)):
            for top_index, top_score, axis in zip(top_indices[i], top_scores[i], axes):
                sample = all_inputs[top_index]
                sample = to_pil_image(sample)
                axis.imshow(sample)
                axis.set_title("{:.4f}".format(top_score))
            canvas.draw()
            plot = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")

            cv2.imshow("k closest", plot)
            user_input = cv2.waitKey(0)
            if user_input != 13:
                break

    # endregion

    @property
    def add_variance_covariance_loss(self) -> bool:
        return (self.variance_lambda > 0.0) or (self.covariance_lambda > 0.0)

    @property
    def logger(self) -> TensorBoardLogger:
        # noinspection PyTypeChecker
        return super(AbstractRepresentationModel, self).logger
