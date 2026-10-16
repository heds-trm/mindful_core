import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Union, Sequence

from mindful_core.models.representation.encoders.multimodal.fusion_module import FusionModule


class MultiModalEncoder(pl.LightningModule):
    def __init__(self,
                 encoders: list[nn.Module],
                 fusion_module: FusionModule,
                 output_module: Optional[nn.Module],
                 modalities: list[str] = None,
                 intermediate_sizes: list[int] = None,
                 modality_dropout: float = None,
                 batch_dropout: bool = False,
                 *args, **kwargs):
        if "output_dimension" in kwargs:
            kwargs.pop("output_dimension")

        super(MultiModalEncoder, self).__init__(*args, **kwargs)
        self.intermediate_sizes = intermediate_sizes

        if modalities:
            if len(modalities) != len(encoders):
                raise ValueError("When `modalities` are provided, the length of `encoders` must match the length "
                                 "of `modalities` but got {} (encoders) and {} (modalities)."
                                 .format(len(encoders), len(encoders)))
        else:
            modalities = list(range(len(encoders)))

        self.modalities = modalities

        self.encoders = encoders
        for modality, encoder in zip(modalities, encoders):
            self.register_module("{}_encoder".format(modality), encoder)

        self.modality_dropout = ModalityDropout(modality_dropout, batch_dropout)
        self.fusion_module = fusion_module
        self.output_module = output_module

        self.check_intermediate_sizes()

    def forward(self,
                inputs: Sequence[torch.Tensor],
                *args,
                output_intermediates=False,
                **kwargs
                ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        # noinspection PyTypeChecker
        inputs, mask = self._unpack_mask(inputs)
        outputs = self.encode_modalities(inputs)

        intermediates = outputs if output_intermediates else None

        outputs = self.modality_dropout(outputs)
        outputs = torch.stack(outputs, dim=1)
        outputs = self.fusion_module(outputs, mask=mask)

        if self.output_module is not None:
            outputs = self.output_module(outputs)

        if output_intermediates:
            return outputs, intermediates

        return outputs

    @staticmethod
    def unpack_mask(inputs: list[torch.Tensor],
                    modality_count: int,
                    ) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(inputs, torch.Tensor):
            raise ValueError("Expected `inputs` to be a list or tuple containing {} (or {})."
                             "Got a single tensor with shape {}.".
                             format(modality_count, modality_count + 1, inputs.shape))

        if len(inputs) == modality_count:
            return inputs, None
        elif len(inputs) == (modality_count + 1):
            return inputs[:-1], inputs[-1]
        else:
            raise ValueError("Expected {} (or {}) inputs, got {}".
                             format(modality_count, modality_count + 1, len(inputs)))

    def _unpack_mask(self,
                     inputs: list[torch.Tensor]
                     ) -> tuple[list[torch.Tensor], Optional[torch.Tensor]]:
        return self.unpack_mask(inputs, self.modality_count)

    def encode_modalities(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [encoder(modality) for (encoder, modality) in zip(self.encoders, inputs)]

    @property
    def fused_size(self) -> int:
        return self.fusion_module.get_fused_size(self.intermediate_sizes)

    def check_intermediate_sizes(self):
        self.fusion_module.get_fused_size(self.intermediate_sizes)

    @property
    def modality_count(self) -> int:
        return len(self.encoders)


class ModalityDropout(nn.Module):
    def __init__(self,
                 dropout: float,
                 drop_whole_batch: bool = False,
                 ) -> None:
        super().__init__()
        self.dropout = dropout
        self.drop_whole_batch = drop_whole_batch

    def forward(self, data: list[torch.Tensor]) -> list[torch.Tensor]:
        if ((self.dropout is None) or
                (self.dropout <= 0.0) or
                not self.training):
            return data

        if self.drop_whole_batch:
            drop_roll = torch.rand(size=(len(data),))
            max_drop_roll = drop_roll.max()
            dropped = (drop_roll < self.dropout) & (drop_roll != max_drop_roll)

            for i in range(len(data)):
                if dropped[i]:
                    data[i] = torch.zeros_like(data[i])

        else:
            batch_size = data[0].size(0)
            device = data[0].device

            drop_roll = torch.rand(size=(len(data), batch_size), device=device)
            max_drop_roll = drop_roll.max(dim=0, keepdim=True)
            kept = (drop_roll >= self.dropout) | (drop_roll == max_drop_roll)
            kept = kept.to(torch.float32)

            for i in range(len(data)):
                kept_ith = torch.reshape(kept[i], shape=[batch_size] + [1] * (data[i].ndim - 1))
                data[i] *= kept_ith

        return data
