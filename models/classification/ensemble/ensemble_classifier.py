import torch
import torch.nn as nn
from pathlib import Path
import os
import warnings
from typing import Any, Iterator

from mindful_core.models.model_output import ClassifierOutput
from mindful_core.models.classification import AbstractClassifier
from mindful_core.models.index import find_mindful_models, load_mindful_models


class EnsembleClassifier(AbstractClassifier):
    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "ensemble_classifier"

    @classmethod
    def module_aliases(cls) -> tuple[str, ...]:
        return ("ensemble",)

    minimum_model_count = 2

    # endregion

    def __init__(self,
                 models_paths: str | Path | list[str] | list[Path],
                 monitor: str,
                 weights: list[float] | torch.Tensor | None = None,
                 accuracies: list[float] | None = None,
                 class_count: int | None = None,
                 optimizer_config: dict[str, dict[str, Any]] | None = None,
                 label_smoothing: float | None = None,
                 confidence_lambda: float | None = None,
                 confidence_corr_lambda: float | None = None,
                 confidence_budget: float | None = None,
                 positive_class: int | None = None,
                 use_focal_loss: bool | None = None,
                 ):
        models_paths = find_mindful_models(models_paths, AbstractClassifier)
        models = load_mindful_models(models_paths, AbstractClassifier, monitor)
        if len(models) < self.minimum_model_count:
            raise ValueError("Not enough model found for an ensemble model, "
                             "found {} but need at least {}.".format(len(models), self.minimum_model_count))
        reference_model = models[0]

        class_count = class_count or reference_model.class_count
        optimizer_config = optimizer_config or reference_model.optimizer_config
        label_smoothing = label_smoothing or reference_model.label_smoothing
        confidence_lambda = confidence_lambda or reference_model.confidence_lambda
        confidence_corr_lambda = confidence_corr_lambda or reference_model.confidence_corr_lambda
        confidence_budget = confidence_budget or reference_model.confidence_budget
        positive_class = reference_model.positive_class if positive_class is None else positive_class
        use_focal_loss = reference_model.use_focal_loss if use_focal_loss is None else use_focal_loss
        super().__init__(class_count=class_count,
                         optimizer_config=optimizer_config,
                         label_smoothing=label_smoothing,
                         confidence_lambda=confidence_lambda,
                         confidence_corr_lambda=confidence_corr_lambda,
                         confidence_budget=confidence_budget,
                         positive_class=positive_class,
                         use_focal_loss=use_focal_loss)

        self.models_paths: list[Path] = models_paths
        self.models: list[AbstractClassifier] = models

        for i, model in enumerate(models):
            self.add_module("model_{:03d}".format(i), model)

        # region Initialise ensemble model weights
        if accuracies is not None:
            if weights is not None:
                warnings.warn("EnsembleClassifier: both `accuracies` and `weights` were provided. "
                              "`accuracies` will overwrite `weights`.")
            accuracies = torch.as_tensor(accuracies, dtype=torch.float32)
            weights = torch.log(accuracies / (1 - accuracies))
        elif weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32)

        if weights is None:
            initial_weights = torch.ones(size=[self.ensemble_weights_size]) / self.ensemble_weights_size
        else:
            initial_weights = weights / weights.sum()
            print("Provided weights: {} (normalized to {})".format(weights.tolist(), initial_weights.tolist()))
        self.ensemble_weights = nn.Parameter(initial_weights)
        # endregion

        if self.yield_confidence:
            self.ensemble_confidence_weights = nn.Parameter(initial_weights.copy_())
        else:
            self.ensemble_confidence_weights = None

    def forward(self, inputs, *args, **kwargs) -> ClassifierOutput:
        with torch.no_grad():
            models_outputs: list[ClassifierOutput] = [model(inputs) for model in self.models]
            models_logits: torch.Tensor = torch.stack([outputs.logits for outputs in models_outputs], dim=0)

        ensemble_weights_shape = [len(self.models)] + [1] * (len(models_logits.shape) - 1)
        ensemble_weights = self.ensemble_weights.view(ensemble_weights_shape)

        logits = (models_logits.detach() * ensemble_weights).sum(dim=0)
        confidence, confidence_threshold = ClassifierOutput.concat_confidence_values(models_outputs)

        intermediate_outputs = models_logits.transpose(0, 1)
        return ClassifierOutput(single_class=self.single_class,
                                logits=logits,
                                intermediate_outputs=intermediate_outputs,
                                confidence=confidence,
                                confidence_threshold=confidence_threshold)

    # region Properties
    @property
    def yield_confidence(self) -> bool:
        return all([model.yield_confidence for model in self.models])

    @property
    def output_features_names(self) -> list[str]:
        common_path = os.path.commonpath(self.models_paths)
        features_names = [os.path.relpath(model_path, common_path) for model_path in self.models_paths]
        return features_names

    @property
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        if self.yield_confidence:
            # noinspection PyTypeChecker
            return [self.ensemble_weights, self.ensemble_confidence_weights]
        else:
            # noinspection PyTypeChecker
            return [self.ensemble_weights]

    @property
    def ensemble_weights_size(self) -> int:
        return len(self.models)
    # endregion
