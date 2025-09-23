import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Any, Union


class AttentionInterface(object):
    def get_attention_modules(self) -> Union["AttentionInterface", list["AttentionInterface"]]:
        return self

    @abstractmethod
    def get_attention_layers(self) -> list[nn.Module]:
        raise NotImplementedError("`get_attention_layers` must be implemented in subclasses.")

    @abstractmethod
    def get_attention_recordings(self, inputs: Any, outputs: Any) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError("`get_attention_weights` must be implemented in subclasses.")

    @property
    @abstractmethod
    def pooling_method(self) -> int | str:
        raise NotImplementedError("`pooling_method` must be implemented in subclasses.")

    @property
    def attention_rank(self) -> int:
        raise NotImplementedError("`attention_rank` must be implemented in subclasses.")

    def format_1d_attention(self, attention_maps: torch.Tensor) -> torch.Tensor:
        # attention_maps shape: [batch_size, n_layers, n_heads, seq_length (+1), seq_length (+1)]
        attention_maps, _ = attention_maps.min(dim=2)
        # attention_maps shape: [batch_size, n_layers, seq_length (+1), seq_length (+1)]
        attention_maps = self.rollout_attention_over_layers(attention_maps)

        # attention_maps shape: [batch_size, seq_length (+1), seq_length (+1)]
        attention_maps = self.reduce_1d_pooling_attention(attention_maps)
        # attention_maps shape: [batch_size, seq_length] or [batch_size, seq_length, seq_length]
        return attention_maps

    @staticmethod
    def rollout_attention_over_layers(attention_maps: torch.Tensor) -> torch.Tensor:
        # attention_maps: [batch_size, n_layers, seq_length, seq_length]
        _, n_layers, seq_length, _ = attention_maps.shape
        identity = torch.eye(seq_length, dtype=attention_maps.dtype, device=attention_maps.device)

        result = torch.clone(identity)
        attention_maps = attention_maps.transpose(1, 0)
        # attention_maps: [n_layers, batch_size, seq_length, seq_length]
        for i in range(n_layers):
            layer_attention = (attention_maps[i] + identity) * 0.5
            layer_attention = layer_attention / layer_attention.sum(dim=-1, keepdim=True)
            result = layer_attention @ result

        # result: [batch_size, seq_length, seq_length]
        return result

    def reduce_1d_pooling_attention(self, attention_maps: torch.Tensor) -> torch.Tensor:
        pooling_method = self.pooling_method

        if pooling_method in [0, 1, "first"]:
            pooling_method = "cls"
        elif pooling_method is None:
            pooling_method = "none"

        if pooling_method == "cls":
            attention_maps = attention_maps[:, 0]
            # attention_maps = attention_maps[:, 0, 1:] if module.use_cls_token else attention_maps[:, 0]
        # else:
        #     attention_maps = attention_maps[:, 1:, 1:] if module.use_cls_token else attention_maps

        return attention_maps
