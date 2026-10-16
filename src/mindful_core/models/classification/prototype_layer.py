##############################################################################
#                                                                            #
# Based on the ProtoPNet ->                                                  #
#                                                                            #
#   CHEN, Chaofan, LI, Oscar, TAO, Daniel, et al.                            #
#   This looks like that: deep learning for interpretable image recognition. #
#   Advances in neural information processing systems, 2019, vol. 32.        #
#                                                                            #
##############################################################################

import torch
import torch.nn as nn
from typing import Literal, Callable

from mindful_core.models.model_output import PrototypeOutput


class PrototypeLayer(nn.Module):
    def __init__(self,
                 input_dimension: int,
                 prototype_count: int,
                 similarity_mode: Literal["log", "linear"] | Callable = "log",
                 class_count: int | None = None
                 ):
        super().__init__()

        self.input_dimension = input_dimension
        self.prototype_count = prototype_count
        self.similarity_mode = similarity_mode
        self.class_count = class_count
        self.class_specific = False if class_count is None else class_count > 1

        if self.class_specific:
            if (self.prototype_count % self.class_count) != 0:
                raise ValueError("When `class_specific` is True, "
                                 "`prototype_count` must be divisible by `class_count`.")

            self.prototypes_class = nn.Parameter(torch.zeros(self.prototype_count, dtype=torch.int32),
                                                 requires_grad=False)
            prototypes_per_class = self.prototype_count // self.class_count
            self.prototypes_per_class = prototypes_per_class
            for label, start in enumerate(range(0, self.prototype_count, prototypes_per_class)):
                self.prototypes_class[start:start + prototypes_per_class] = label
        else:
            self.prototypes_per_class = None
            self.prototypes_class = None

        self.prototypes = nn.Parameter(torch.rand(self.prototypes_shape) / self.prototype_count, requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> PrototypeOutput:
        distances = self.representation_to_prototype_distance(inputs)
        similarities = self.distance_to_similarity(distances)
        return PrototypeOutput(distances, similarities)

    def representation_to_prototype_distance(self, representation: torch.Tensor) -> torch.Tensor:
        representation = representation.unsqueeze(1)
        prototypes = self.prototypes.unsqueeze(0)
        l2_distance = torch.sqrt(torch.square(representation - prototypes).sum(2))
        return l2_distance

    def distance_to_similarity(self, distances: torch.Tensor) -> torch.Tensor:
        if self.similarity_mode == "log":
            return torch.log((distances + 1) / (distances + 1e-5))
        elif self.similarity_mode == "linear":
            return -distances
        else:
            return self.similarity_mode(distances)

    def compute_class_specific_loss(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = labels.shape[0]
        labels = labels.to(torch.int32)
        # - distances: [batch_size, prototype_count]
        matching_class_mask = self.prototypes_class.unsqueeze(0) == labels.unsqueeze(1)
        # - class_mask: [batch_size, prototype_count]

        # region (minimize) Distance to closest prototype of given class
        matching_class_distances = distances[matching_class_mask].view(batch_size, self.prototypes_per_class)
        # - matching_class_distances: [batch_size, prototypes_per_class]
        matching_class_min_distances, _ = matching_class_distances.min(dim=1)
        # - matching_class_min_distances: [batch_size]
        matching_class_loss = matching_class_min_distances.mean()
        # endregion

        # region (maximize) Distance to prototype of other classes
        prototypes_for_other_classes = self.prototypes_per_class * (self.class_count - 1)
        other_class_distances = distances[~matching_class_mask].view(batch_size, prototypes_for_other_classes)
        # - other_class_distances: [batch_size, prototypes_for_other_classes]
        max_expected_distance = self.prototype_count
        other_class_loss = torch.relu(max_expected_distance - other_class_distances).mean()
        # endregion

        return matching_class_loss + other_class_loss

    @staticmethod
    def compute_class_agnostic_loss(distances: torch.Tensor) -> torch.Tensor:
        # - distances: [batch_size, prototype_count]
        min_distances, _ = distances.min(dim=1)
        # - min_distance: [batch_size]
        loss = min_distances.mean()
        return loss

    @property
    def prototypes_shape(self) -> tuple[int, int]:
        return self.prototype_count, self.input_dimension

    @staticmethod
    def model_has_prototype_layer(model: nn.Module):
        for module in model.modules():
            if isinstance(module, PrototypeLayer):
                return True

        return False
