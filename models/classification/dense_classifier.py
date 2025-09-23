import torch
import torch.nn as nn
from typing import Sequence

from models.model_output import ClassifierOutput


class DenseClassifier(nn.Module):
    def __init__(self,
                 input_dimension: int,
                 features: Sequence[int],
                 class_count: int,
                 yield_confidence: bool = False
                 ):
        super(DenseClassifier, self).__init__()
        self.input_dimension = input_dimension
        self.features = features
        self.class_count = class_count
        self.yield_confidence = yield_confidence

        # region Intermediate layers
        self.intermediate_layers = []
        features = [input_dimension, *features]
        for i in range(len(features) - 1):
            linear_layer = nn.Linear(features[i], features[i + 1])
            self.add_module(name="linear_{}".format(i), module=linear_layer)
            activation_layer = nn.ReLU()
            self.add_module(name="activation_{}".format(i), module=activation_layer)
            self.intermediate_layers += [linear_layer, activation_layer]
        # endregion

        # region Output layer (logits)
        self.output_layer = nn.Linear(features[-1], class_count)
        self.add_module(name="output_layer", module=self.output_layer)
        # endregion

        # region Confidence layer (optional)
        if self.yield_confidence:
            self.confidence_layer = nn.Linear(features[-1], 1)
            self.add_module(name="confidence_layer", module=self.confidence_layer)
        else:
            self.confidence_layer = None
        # endregion

    def forward(self, inputs: torch.Tensor) -> ClassifierOutput:
        for layer in self.intermediate_layers:
            inputs = layer(inputs)
        logits = self.output_layer(inputs)

        if self.yield_confidence:
            confidence = self.confidence_layer(inputs).squeeze(-1)
        else:
            confidence = None

        return ClassifierOutput(single_class=self.single_class,
                                logits=logits,
                                confidence=confidence,
                                confidence_threshold=None)

    @property
    def single_class(self) -> bool:
        return self.class_count == 1
