import torch
import torch.nn as nn
from einops import repeat
from typing import Union, Literal

POOLING_OPTIONS = int | Literal["cls", "first", "last", "mean", "max", "mul", "prod", "none"] | None


class AddClsToken(nn.Module):
    def __init__(self, hidden_size: int):
        super(AddClsToken, self).__init__()
        self.hidden_size = hidden_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self,
                data: Union[torch.Tensor, list[torch.Tensor]],
                mask: torch.Tensor = None
                ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size = data[0].shape[0] if isinstance(data, list) else data.shape[0]
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)

        if isinstance(data, list):
            cls_token = cls_token.squeeze(dim=1)
            data = torch.stack([cls_token] + data, dim=1)
        else:
            data = torch.concat([cls_token, data], dim=1)

        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device)
            mask = torch.concat([cls_mask, mask], dim=1)
            return data, mask

        return data


class AddPositionalEmbeddings(nn.Module):
    def __init__(self, max_length: int, hidden_size: int):
        super(AddPositionalEmbeddings, self).__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.weights = nn.Parameter(torch.rand(max_length, hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = inputs.shape
        if length < self.max_length:
            weights = self.weights[:length]
        else:
            weights = self.weights
        weights = weights.expand(batch_size, -1, -1)
        return inputs + weights


class TransformerPooling(nn.Module):
    def __init__(self,
                 pooling: POOLING_OPTIONS,
                 dim: int = 1,
                 ) -> None:
        super(TransformerPooling, self).__init__()
        self.pooling = pooling
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (self.pooling is None) or (self.pooling == "none"):
            return inputs

        elif self.pooling in ["first", "cls", "last", 0, 1, -1]:
            i = -1 if self.pooling in ["last", -1] else 0
            if self.dim == 0:
                return inputs[i]
            elif self.dim == 1:
                return inputs[:, i]
            elif self.dim == 2:
                return inputs[:, :, i]
            elif self.dim == -1:
                return inputs[..., i]

            dim = self.dim if self.dim >= 0 else len(inputs.shape) + self.dim
            pooling_slice = [slice(None) if j == dim else slice(1) for j in range(len(inputs.shape))]
            inputs = inputs[pooling_slice].squeeze(self.dim)
            return inputs

        elif self.pooling in ["mean", "avg", "average"]:
            return inputs.mean(dim=self.dim)

        elif self.pooling == "max":
            _max = inputs.max(dim=self.dim)
            if not isinstance(_max, torch.Tensor):
                _max, _ = _max
            return _max

        elif self.pooling in ["mul", "prod"]:
            return inputs.prod(dim=self.dim)

        elif isinstance(self.pooling, int):
            if self.dim == 0:
                return inputs[:self.pooling]
            elif self.dim == 1:
                return inputs[:, :self.pooling]
            elif self.dim == 2:
                return inputs[:, :, :self.pooling]
            elif self.dim == -1:
                return inputs[..., :self.pooling]

            dim = self.dim if self.dim >= 0 else len(inputs.shape) + self.dim
            pooling_slice = [slice(None) if i == dim else slice(self.pooling) for i in range(len(inputs.shape))]
            return inputs[pooling_slice]

        raise NotImplementedError("Unknown pooling method `{}`".format(self.pooling))
