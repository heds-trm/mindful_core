import torch
from torch.nn import Embedding, Linear, Sequential, ReLU
import pytorch_lightning as pl


class CategoricalEncoder(pl.LightningModule):
    def __init__(self,
                 categories_sizes: list[int],
                 categories_hidden_sizes: list[int] | int,
                 linear_hidden_sizes: list[int] | int,
                 output_dimension: int,
                 add_missing_token: bool = True,
                 *args,
                 **kwargs):
        super(CategoricalEncoder, self).__init__(*args, **kwargs)

        if len(categories_sizes) == 0:
            raise ValueError("You did not provide any category size to CategoricalEncoder.")

        self.categories_sizes = categories_sizes
        self.hidden_sizes = categories_hidden_sizes
        self.output_dimension = output_dimension
        self.add_missing_token = add_missing_token

        # region Embedding layers
        if isinstance(categories_hidden_sizes, int):
            categories_hidden_sizes = [categories_hidden_sizes] * len(categories_sizes)

        input_sizes = categories_sizes
        if add_missing_token:
            input_sizes = [size + 1 for size in input_sizes]

        self.embedding_layers = [
            Embedding(input_size, hidden_size) for (input_size, hidden_size) in
            zip(input_sizes, categories_hidden_sizes)
        ]
        for i, embedding_layer in enumerate(self.embedding_layers):
            self.register_module("EmbeddingLayer_{}".format(i), embedding_layer)
        # endregion

        # region Linear layers (intermediate)
        total_hidden_size = sum(categories_hidden_sizes)

        if isinstance(linear_hidden_sizes, int):
            linear_hidden_sizes = [linear_hidden_sizes]
        linear_hidden_sizes.insert(0, total_hidden_size)

        if len(linear_hidden_sizes) > 1:
            intermediate_layers = []
            for i in range(len(linear_hidden_sizes) - 1):
                in_features = linear_hidden_sizes[i]
                out_features = linear_hidden_sizes[i + 1]
                intermediate_layers.append(Linear(in_features, out_features))
                intermediate_layers.append(ReLU())
            self.intermediate_layers = Sequential(*intermediate_layers)
        else:
            self.intermediate_layers = None
        # endregion

        # region Output layer
        self.output_layer = Linear(linear_hidden_sizes[-1], output_dimension)
        # endregion

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        outputs = self.unbind_categories(inputs)
        outputs = torch.concat([embedding_layer(unbound_input)
                                for (embedding_layer, unbound_input)
                                in zip(self.embedding_layers, outputs)], dim=-1)
        if self.intermediate_layers is not None:
            outputs = self.intermediate_layers(outputs)
        outputs = self.output_layer(outputs)
        return outputs

    def unbind_categories(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        # Expected_inputs : [batch_size, category_count] or [batch_size]
        # Will squeeze all extra dimensions if it allows to match the pattern
        if len(self.embedding_layers) == 1:
            if len(inputs.shape) != 1:
                if inputs.numel() == inputs.size(0):
                    inputs = inputs.squeeze()
                else:
                    raise ValueError("Expected inputs to have 1 dimension (batch_size), got {}.".format(inputs.shape))
            return [inputs]
        else:
            if len(inputs.shape) != 2:
                batch_size = inputs.size(0)
                if inputs.numel() == (batch_size * self.embedding_layers):
                    inputs = inputs.squeeze()
                else:
                    raise ValueError("Expected inputs to have 2 dimensions (batch_size and category count), got {}."
                                     .format(len(inputs.shape)))
            if inputs.size(1) != len(self.embedding_layers):
                raise ValueError("Expected {} categories, got {}.".format(len(self.embedding_layers), inputs.size(1)))
            return list(torch.unbind(inputs, dim=1))
