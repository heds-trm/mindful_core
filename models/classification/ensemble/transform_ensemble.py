import torch
from pathlib import Path
from typing import Any

from data.transforms.pipeline import TransformConfig, SerializableTransform
from models.model_output import ClassifierOutput
from models.classification.abstract_classifier import AbstractClassifier
from models.classification.ensemble.ensemble_classifier import EnsembleClassifier
from utils.misc import generate_binary_combinations


class TransformEnsembleClassifier(EnsembleClassifier):
    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "transform_ensemble_classifier"

    @classmethod
    def module_aliases(cls) -> tuple[str, ...]:
        return ("transform_ensemble",)

    minimum_model_count = 1

    # endregion

    def __init__(self,
                 model_path: str | Path,
                 transforms: list[tuple[int | list[int], TransformConfig]],
                 monitor: str,
                 class_count: int | None = None,
                 optimizer_config: dict[str, dict[str, Any]] | None = None,
                 label_smoothing: float | None = None,
                 confidence_lambda: float | None = None,
                 confidence_corr_lambda: float | None = None,
                 confidence_budget: float | None = None,
                 positive_class: int | None = None,
                 use_focal_loss: bool | None = None,
                 ) -> None:
        super().__init__(models_paths=[model_path],
                         monitor=monitor,
                         class_count=class_count,
                         optimizer_config=optimizer_config,
                         label_smoothing=label_smoothing,
                         confidence_lambda=confidence_lambda,
                         confidence_corr_lambda=confidence_corr_lambda,
                         confidence_budget=confidence_budget,
                         positive_class=positive_class,
                         use_focal_loss=use_focal_loss)

        transform_configs = transforms
        transforms = []
        for modality_index, transform_config in transform_configs:
            transform = SerializableTransform.deserialize(transform_config["type"], transform_config["parameters"])
            transforms.append((modality_index, transform))

        self.transforms = transforms

    def forward(self, inputs, *args, **kwargs):
        with torch.no_grad():
            models_outputs: list[ClassifierOutput] = [self.model(self.apply_selected_transforms(inputs, combination))
                                                      for combination
                                                      in generate_binary_combinations(length=len(self.transforms))]
            models_logits: torch.Tensor = torch.stack([outputs.logits for outputs in models_outputs], dim=0)

    def apply_selected_transforms(self,
                                  inputs: torch.Tensor | tuple[torch.Tensor, ...],
                                  combination: list[bool]
                                  ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        for apply_transform, (modality_index, transform) in zip(combination, self.transforms):
            if apply_transform:
                inputs = transform(inputs)
            return inputs

    @property
    def model(self) -> AbstractClassifier:
        return self.models[0]
