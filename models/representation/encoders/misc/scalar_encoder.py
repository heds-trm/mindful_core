import torch
from torch.nn import Linear, Sequential, ReLU
import pytorch_lightning as pl


class ScalarEncoder(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list[int] | int,
                 output_dimension: int,
                 *args, **kwargs):
        super(ScalarEncoder, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_dimension = output_dimension

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_sizes.insert(0, input_size)

        if len(hidden_sizes) > 1:
            intermediate_layers = []
            for i in range(len(hidden_sizes) - 1):
                in_features = hidden_sizes[i]
                out_features = hidden_sizes[i + 1]
                intermediate_layers.append(Linear(in_features, out_features))
                intermediate_layers.append(ReLU())
            self.intermediate_layers = Sequential(*intermediate_layers)
        else:
            self.intermediate_layers = None

        self.output_layer = Linear(hidden_sizes[-1], output_dimension)

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        outputs = inputs
        if self.intermediate_layers is not None:
            outputs = self.intermediate_layers(outputs)
        outputs = self.output_layer(outputs)
        return outputs
