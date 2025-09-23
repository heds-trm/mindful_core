import torch
import torch.nn as nn
from typing import Any, Iterator

from data.subset_id import SubsetID
from models.model_output import ClassifierOutput
from models.classification import AbstractClassifier, DenseClassifier
from models.representation.abstract_representation_model import AbstractRepresentationModel, RepresentationOutput
from models.loss_aggregator import LossAggregator


class TargetModel(AbstractClassifier):
    """
        todo: write documentation
    """

    # region Class/Subclass methods
    @classmethod
    def required_sub_modules(cls) -> list[str]:
        return ["representation_model"]

    @classmethod
    def module_identifier(cls) -> str:
        return "target"

    @classmethod
    def module_aliases(cls) -> tuple[str, ...]:
        return ("mlp", )

    # endregion
    def __init__(self,
                 representation_model: AbstractRepresentationModel,
                 classifier_config: dict[str, Any],
                 classifier_features: list[int] = (),
                 freeze_representation_model: bool = False,
                 label_smoothing=0.0,
                 confidence_lambda=1e-1,
                 confidence_corr_lambda=1e-1,
                 # good default values are in [0.1, 1.0]
                 confidence_budget=None,
                 positive_class=-1,
                 train_intermediates: bool = False,
                 use_focal_loss=False,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 **kwargs):
        super(TargetModel, self).__init__(class_count=classifier_config["class_count"],
                                          optimizer_config=optimizer_config,
                                          label_smoothing=label_smoothing,
                                          confidence_lambda=confidence_lambda,
                                          confidence_corr_lambda=confidence_corr_lambda,
                                          confidence_budget=confidence_budget,
                                          positive_class=positive_class,
                                          use_focal_loss=use_focal_loss,
                                          **kwargs)
        self.representation_model = representation_model
        self.freeze_representation_model = freeze_representation_model
        self.train_intermediates = train_intermediates
        self.setup_representation_model()

        self.classifier_features = classifier_features
        self.classifier = DenseClassifier(input_dimension=self.classifier_input_dimension, **classifier_config)

    def setup_representation_model(self, freeze: bool = None):
        if freeze is None:
            freeze = self.freeze_representation_model

        if freeze:
            self.representation_model.freeze()
            self.representation_model.eval()
        else:
            self.representation_model.unfreeze()
            self.representation_model.train()

    # region Forward
    def forward(self, inputs, *args, **kwargs) -> ClassifierOutput:
        rep_outputs: RepresentationOutput = self.representation_model(inputs, *args, flatten=True, **kwargs)
        outputs = self.representation_to_logits(rep_outputs.representations)

        intermediate_logits = self.map_intermediate(rep_outputs.intermediate_outputs)
        outputs.intermediate_outputs = intermediate_logits

        return outputs

    def representation_to_logits(self, representation: torch.Tensor) -> ClassifierOutput:
        if self.freeze_representation_model:
            representation = representation.detach()

        outputs: ClassifierOutput = self.classifier(representation)
        outputs.confidence_threshold = self.confidence_threshold
        return outputs

    # endregion

    def compute_output_losses(self,
                              outputs: ClassifierOutput,
                              labels: torch.Tensor,
                              subset_id: SubsetID) -> LossAggregator:
        super(TargetModel, self).compute_output_losses(outputs, labels, subset_id)

        if self.train_intermediates:
            self.compute_intermediates_loss(outputs.intermediate_outputs, labels, subset_id)

        return self.loss_aggregator

    def compute_intermediates_loss(self,
                                   intermediate_outputs: list[torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
                                   labels: torch.Tensor,
                                   subset_id: SubsetID) -> LossAggregator:
        intermediates_losses = [self.compute_losses_for_intermediate(intermediate_output, labels, subset_id)
                                for intermediate_output in intermediate_outputs]
        intermediates_loss = torch.stack(intermediates_losses).mean()

        self.loss_aggregator.add_loss(intermediates_loss, "intermediates_loss", 1e-1)
        return self.loss_aggregator

    def compute_losses_for_intermediate(self,
                                        intermediate_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
                                        labels: torch.Tensor,
                                        subset_id: SubsetID
                                        ) -> torch.Tensor:
        raise NotImplementedError

    # region Intermediate
    @staticmethod
    def unpack_intermediate(outputs: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
                            ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if isinstance(outputs, tuple):
            # noinspection PyTypeChecker
            return outputs
        else:
            return outputs, None

    def map_intermediate(self,
                         intermediate_representations: list[torch.Tensor] | None,
                         ) -> list[ClassifierOutput] | None:
        if intermediate_representations is None:
            return None

        return [self.representation_to_logits(tensor) for tensor in intermediate_representations]

    # endregion

    # region Properties
    @property
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.classifier.parameters() if self.freeze_representation_model else self.parameters()

    @property
    def yield_confidence(self) -> bool:
        return self.classifier.yield_confidence

    @property
    def intermediate_size(self) -> int:
        return self.representation_model.output_dimension

    @property
    def intermediate_dimension(self) -> int:
        return self.intermediate_size

    @property
    def classifier_input_dimension(self) -> int:
        return self.intermediate_size

    # endregion
