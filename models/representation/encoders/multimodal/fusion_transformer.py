import torch
import torch.nn as nn
import einops
from typing import Literal

from mindful_core.utils.transformers import TransformerPooling, AddClsToken, AddPositionalEmbeddings, POOLING_OPTIONS
from mindful_core.models.representation.encoders.multimodal.fusion_module import FusionModule
from mindful_core.models.attention_interface import AttentionInterface


class FusionTransformer(FusionModule, AttentionInterface):
    def __init__(self,
                 input_size: int,
                 head_count: int,
                 layer_count: int,
                 add_cls_token=False,
                 add_position_embeddings=True,
                 pooling: POOLING_OPTIONS = "auto",
                 project_output: bool = True,
                 output_dimension: int = None,
                 pre_norm_outputs: bool = False,
                 pre_activation: str | None = "ReLU",
                 output_activation: Literal["Tanh"] | None = None,
                 modality_count=None):
        super().__init__(default_mask_replacement=0.0)

        output_dimension = input_size if (project_output and (output_dimension is None)) else output_dimension
        if (modality_count is None) and add_position_embeddings:
            raise ValueError("When adding positional embeddings, `modality_count` must be known.")

        self.input_size = input_size
        self.head_count = head_count
        self.layer_count = layer_count
        self.use_cls_token = add_cls_token
        self.use_position_embeddings = add_position_embeddings

        self.project_output = project_output
        self.output_dimension = output_dimension
        self.pre_norm_outputs = pre_norm_outputs
        self.pre_activation = pre_activation
        self.output_activation = output_activation

        self.modality_count = modality_count

        # region Class token / Positional embeddings
        self.cls_token = AddClsToken(input_size) if self.use_cls_token else None
        if self.use_position_embeddings:
            embeddings_count = modality_count + 1 if self.use_cls_token else modality_count
            self.pos_embeddings = AddPositionalEmbeddings(embeddings_count, input_size)
        else:
            self.pos_embeddings = None
        # endregion

        # region Transformer
        encoder_layer = FusionTransformerEncoderLayer(d_model=input_size, nhead=head_count, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=layer_count)
        # endregion

        # region Pooling
        if pooling == "auto":
            pooling = "cls" if self.use_cls_token else "mean"
        self.pooling = TransformerPooling(pooling)
        # endregion

        # region Output projection
        if self.project_output:
            self.projector = FusionProjector(in_features=input_size, out_features=output_dimension,
                                             pre_norm=pre_norm_outputs, pre_activation=pre_activation,
                                             output_activation=output_activation)
        else:
            self.projector = None
        # endregion

    def fuse(self,
             inputs: list[torch.Tensor] | torch.Tensor,
             mask=None,
             weights=None,
             *args,
             **kwargs
             ) -> torch.Tensor:
        inputs, mask = self.prepare_inputs_and_mask(inputs, mask)

        inputs = self.transformer(inputs, mask=mask)
        inputs = self.pooling(inputs)

        if self.project_output:
            inputs = self.projector(inputs)

        return inputs

    # region Prepare inputs/mask
    def prepare_inputs_and_mask(self,
                                inputs: list[torch.Tensor],
                                mask: torch.Tensor = None
                                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_cls_token:
            inputs = self.cls_token(inputs, mask)
            if mask is not None:
                inputs, mask = inputs

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.stack(inputs, dim=1)

        if self.use_position_embeddings:
            inputs = self.pos_embeddings(inputs)

        if mask is not None:
            mask = self.repeat_and_correct_mask(mask)

        return inputs, mask

    def repeat_and_correct_mask(self, mask: torch.Tensor) -> torch.Tensor:
        src_length = mask.size(1)
        # mask = mask.unsqueeze(-2).repeat(self.head_count, src_length, 1)
        mask = einops.repeat(mask, "batch src_length -> (batch head_count) tgt_length src_length",
                             head_count=self.head_count, tgt_length=src_length)
        if mask.dtype == torch.bool:
            # Torch uses the mask in the opposite way when mask dtype is bool for some reason \o/
            mask = torch.logical_not(mask)
        return mask

    # endregion

    # region AttentionInterface
    def get_attention_layers(self) -> list[nn.Module]:
        attention_layers = [module for module in self.transformer.layers.modules()
                            if isinstance(module, FusionTransformerEncoderLayer)]
        return attention_layers

    def get_attention_recordings(self,
                                 inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                 outputs: tuple[torch.Tensor, torch.Tensor]
                                 ) -> torch.Tensor | list[torch.Tensor]:
        inputs, _, _ = inputs
        _, attention_weights = outputs
        return [attention_weights, inputs]

    @property
    def pooling_method(self) -> int | str:
        return self.pooling.pooling

    @property
    def attention_rank(self) -> int:
        return 1

    # endregion


class FusionTransformerEncoderLayer(nn.TransformerEncoderLayer):
    # Switching the `need_weights` flag to True to allow recording through forward hooks
    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: torch.Tensor | None,
                  key_padding_mask: torch.Tensor | None,
                  is_causal: bool = False) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True,
                           average_attn_weights=False,
                           is_causal=is_causal)[0]
        return self.dropout1(x)


class FusionProjector(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 pre_norm: bool = False,
                 pre_activation: Literal["ReLU"] | None = None,
                 output_activation: Literal["Tanh"] | None = None,
                 ) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.pre_norm = nn.LayerNorm(in_features) if pre_norm else None
        self.pre_activation = nn.ReLU() if pre_activation == "ReLU" else None
        self.output_activation = nn.Tanh() if output_activation == "Tanh" else None

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is not None:
            data = self.pre_norm(data)

        if self.pre_activation is not None:
            data = self.pre_activation(data)

        data = self.linear(data)

        if self.output_activation is not None:
            data = self.output_activation(data)

        return data
