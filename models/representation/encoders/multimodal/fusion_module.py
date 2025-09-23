import torch
import torch.nn as nn
from functools import reduce
from abc import abstractmethod
from typing import Literal

from models.representation.encoders.misc.modalities_attention_add import ModalitiesAttentionAdd
from utils.transformers import TransformerPooling


class FusionModule(nn.Module):
    def __init__(self, default_mask_replacement: float):
        super(FusionModule, self).__init__()
        self.default_mask_replacement = default_mask_replacement

    def forward(self, inputs: torch.Tensor, mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        if mask is not None:
            inputs = self.apply_mask(inputs, mask)

        if weights is not None:
            inputs = self.apply_weights(inputs, weights)

        return self.fuse(inputs, mask=mask, weights=weights, *args, **kwargs)

    @abstractmethod
    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def apply_weights(inputs: list[torch.Tensor] | torch.Tensor, weights: torch.Tensor):
        if len(inputs) != weights.size(-1):
            raise ValueError("Expected the last dimension of the weights vector to match the number of available "
                             "modalities. Got {} modalities and weights' last dimension was {}."
                             .format(len(inputs), weights.size(-1)))

        for modality_index in range(len(inputs)):
            inputs[modality_index] *= weights[modality_index]

        return inputs

    def apply_mask(self, inputs: torch.Tensor, mask: torch.Tensor):
        if inputs.size(1) != mask.size(1):
            raise ValueError("Expected the last dimension of the mask vector to match the number of available "
                             "modalities. Got {} modalities and mask's length was {}."
                             .format(inputs.size(1), mask.size(1)))

        if mask.dtype != torch.bool:
            mask = mask.greater(0.5)

        default_mask_replacement = torch.as_tensor(self.default_mask_replacement, dtype=torch.float32,
                                                   device=inputs[0].device)
        
        return torch.where(mask.unsqueeze(-1), inputs, default_mask_replacement)

    @classmethod
    def get_fused_size(cls, input_sizes: list[int]) -> int:
        for size in input_sizes[1:]:
            if size != input_sizes[0]:
                raise RuntimeError("All input sizes must be equal, got {}.".format(input_sizes))

        return input_sizes[0]


class ConcatFusion(FusionModule):
    def __init__(self, dim=-1):
        super(ConcatFusion, self).__init__(default_mask_replacement=0.0)
        self.dim = dim

    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        return torch.concat(inputs, dim=self.dim)

    @classmethod
    def get_fused_size(cls, input_sizes: list[int]) -> int:
        return sum(input_sizes)


class AdditiveFusion(FusionModule):
    def __init__(self, boost_for_masked_modalities: bool):
        super(AdditiveFusion, self).__init__(default_mask_replacement=0.0)
        self.boost_for_masked_modalities = boost_for_masked_modalities

    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        fused_inputs = torch.zeros_like(inputs[0])
        # noinspection PyTypeChecker
        fused_inputs = reduce(torch.Tensor.add_, inputs, fused_inputs)

        if (mask is not None) and self.boost_for_masked_modalities:
            if mask.dtype != torch.float:
                mask = mask.to(torch.float)
            mask_mean = mask.mean(dim=-1, keepdim=True)
            fused_inputs /= mask_mean

        return fused_inputs


class MeanFusion(FusionModule):
    def __init__(self):
        super(MeanFusion, self).__init__(default_mask_replacement=0.0)

    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        fused_inputs = torch.zeros_like(inputs[0])
        # noinspection PyTypeChecker
        fused_inputs = reduce(torch.Tensor.add_, inputs, fused_inputs)

        if mask is not None:
            if mask.dtype != torch.float:
                mask = mask.to(torch.float)
            modality_count = mask.sum(dim=-1, keepdim=True)
        else:
            modality_count = len(fused_inputs)

        return fused_inputs / modality_count


class MaxFusion(FusionModule):
    def __init__(self):
        super(MaxFusion, self).__init__(default_mask_replacement=0.0)

    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        fused_inputs, _ = torch.stack(inputs, dim=1).max(dim=1)
        return fused_inputs


class MultiplyFusion(FusionModule):
    def __init__(self):
        super(MultiplyFusion, self).__init__(default_mask_replacement=1.0)

    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        fused_inputs = torch.ones_like(inputs[0])
        # noinspection PyTypeChecker
        fused_inputs = reduce(torch.Tensor.mul_, inputs, fused_inputs)
        return fused_inputs


class AttentionAddFusion(FusionModule):
    def __init__(self, modality_count: int, temperature=1.0):
        super(AttentionAddFusion, self).__init__(default_mask_replacement=0.0)
        self.attention_add = ModalitiesAttentionAdd(modality_count=modality_count,
                                                    fused_size=self.fused_size,
                                                    temperature=temperature)

    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        fused_inputs = torch.stack(inputs, dim=1)
        fused_inputs = self.attention_add(fused_inputs, mask=mask)
        return fused_inputs


class MultiHeadAttentionFusion(FusionModule):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadAttentionFusion, self).__init__(default_mask_replacement=0.0)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        fused_inputs = torch.stack(inputs, dim=1)
        return self.multihead_attention(fused_inputs, mask=mask)
    

class LSTMFusion(FusionModule):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int, 
                 learn_initial_state: bool,
                 pooling: int | Literal["cls", "first", "last", "mean", "max", "mul", "prod", "none"] | None = "last",
                 num_layers: int = 1, 
                 bias=True,
                 dropout=0.0,
                 bidirectional=False,
                 proj_size=0,
                 ):
        super().__init__(default_mask_replacement=0.0)
        self.learn_initial_state = learn_initial_state
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            bias=bias,
                            batch_first=True,
                            dropout=dropout, 
                            bidirectional=bidirectional,
                            proj_size=proj_size)
        
        if learn_initial_state:
            initial_h0 = torch.randn(size=self.initial_state_shape)
            initial_c0 = torch.randn(size=self.initial_state_shape)
            self.initial_hidden_state = nn.Parameter(initial_h0)
            self.initial_cell_state = nn.Parameter(initial_c0)
        else:
            self.initial_hidden_state = torch.zeros(self.initial_state_shape, dtype=torch.float32)
            self.initial_cell_state = torch.zeros(self.initial_state_shape, dtype=torch.float32)

        self.pooling = TransformerPooling(pooling)
        
    def fuse(self, inputs: list[torch.Tensor] | torch.Tensor, mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            inputs: torch.Tensor = torch.stack(inputs, dim=1)
        batch_size = inputs.shape[0]

        initial_hidden_state = self.initial_hidden_state.repeat(1, batch_size, 1)
        initial_cell_state = self.initial_cell_state.repeat(1, batch_size, 1)
        initial_state = (initial_hidden_state, initial_cell_state)

        outputs, _ = self.lstm(inputs, initial_state)
        outputs = self.pooling(outputs)
        return outputs
    
    @property
    def bidirectional(self) -> bool:
        return self.lstm.bidirectional

    @property
    def bidirectional_size_factor(self) -> int:
        return 2 if self.bidirectional else 1
    
    @property
    def num_layers(self) -> int:
        return self.lstm.num_layers
    
    @property
    def output_size(self) -> int:
        return self.lstm.proj_size if self.lstm.proj_size > 0 else self.lstm.hidden_size
    
    @property
    def initial_state_shape(self) -> torch.Size:
        return torch.Size([self.bidirectional_size_factor * self.num_layers, 1, self.output_size])
        

class StackFusion(FusionModule):
    def __init__(self, default_mask_replacement=0.0):
        super(StackFusion, self).__init__(default_mask_replacement=default_mask_replacement)

    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        fused_inputs = torch.stack(inputs, dim=1)
        return fused_inputs


class HiNetFusion(FusionModule):
    """
    See `Hi-Net: Hybrid-Fusion Network for Multi-Modal MR Image Synthesis`: https://doi.org/10.1109/TMI.2020.2975344
    """

    def __init__(self, fusion_modules: list[FusionModule]):
        super(HiNetFusion, self).__init__(default_mask_replacement=0.0)
        self.fusion_modules = fusion_modules

    def fuse(self, inputs: list[torch.Tensor], mask=None, weights=None, *args, **kwargs) -> torch.Tensor:
        inputs = [module(inputs, mask=mask, weights=weights, *args, **kwargs) for module in self.fusion_modules]
        return torch.concat(inputs, dim=-1)

    @classmethod
    def get_fused_size(cls, input_sizes: list[int]) -> int:
        return sum(input_sizes)
