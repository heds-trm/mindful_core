import math
import torch
import torch.nn as nn
from monai.networks.nets import ViT
from monai.networks.nets.swin_unetr import BasicLayer as BasicSwinLayer
# noinspection PyProtectedMember
from monai.networks.nets.swin_unetr import MERGING_MODE, UnetrBasicBlock
from monai.networks.blocks import PatchEmbed
from monai.utils import ensure_tuple_rep, look_up_option
import numpy as np
from typing import Sequence, Literal

from utils.transformers import TransformerPooling, AddClsToken


class ViTEncoder(ViT):
    def __init__(self,
                 in_channels: int,
                 img_size: Sequence[int] | int,
                 patch_size: Sequence[int] | int,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 output_dimension: int = 512,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 proj_type: str = "conv",
                 dropout_rate: float = 0.0,
                 norm_outputs: bool = False,
                 cut_init: float | None = None,
                 spatial_dims: int = 3,
                 pooling: int | Literal["cls", "first", "mean", "max", "mul", "prod", "none"] | None = "first",
                 # **kwargs
                 ):
        project_outputs = (hidden_size != output_dimension) or (cut_init is not None)
        super(ViTEncoder, self).__init__(in_channels=in_channels,
                                         img_size=img_size,
                                         patch_size=patch_size,
                                         hidden_size=hidden_size,
                                         mlp_dim=mlp_dim,
                                         num_classes=output_dimension,
                                         num_layers=num_layers,
                                         num_heads=num_heads,
                                         proj_type=proj_type,
                                         classification=False,
                                         dropout_rate=dropout_rate,
                                         spatial_dims=spatial_dims,
                                         post_activation="Linear",
                                         )
        self.spatial_dims = spatial_dims
        self.img_size = ensure_tuple_rep(img_size, dim=spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, dim=spatial_dims)
        self.norm_outputs = norm_outputs
        self.cut_init = cut_init
        self.project_outputs = project_outputs
        self.output_dimension = output_dimension
        self.pooling = TransformerPooling(pooling)

        if project_outputs:
            self.classification_head = nn.Linear(hidden_size, output_dimension)
            if cut_init is not None:
                bound = 1.0 / math.sqrt(output_dimension) / cut_init
                nn.init.uniform_(self.classification_head.weight, -bound, bound)
        else:
            self.classification_head = None

        self.cls_token = AddClsToken(hidden_size)
        self.patch_count = tuple([image_dim // patch_dim for image_dim, patch_dim
                                  in zip(self.img_size, self.patch_size)])
        self.total_patch_count = np.prod(self.patch_count)

    def forward(self, inputs, *args, **kwargs):
        outputs = self.patch_embedding(inputs)
        outputs = self.cls_token(outputs)

        for block in self.blocks:
            outputs = block(outputs)

        outputs = self.pooling(outputs)

        if self.norm_outputs:
            outputs = self.norm(outputs)

        if self.project_outputs:
            outputs = self.classification_head(outputs)

        return outputs


class SwinEncoder(nn.Module):
    """
    Hybrid version of the Swin Transformer (between the original and MONAI's)
    Swin Transformer based on: "Liu et al., Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>" https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            in_channels: int = 1,
            window_size: Sequence[int] | int = 7,
            patch_size: Sequence[int] | int = 4,
            hidden_size: int = 96,
            output_dimension: int = 512,
            depths: Sequence[int] = (2, 2, 6, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            pooling: int | Literal["cls", "first", "mean", "max", "mul", "prod", "none"] | None = "mean",
            norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
            norm_outputs: bool = False,
            norm_patch: bool = False,
            cut_init: float | None = None,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            downsample="merging",
            use_v2=False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            norm_outputs: add normalization after swin layers.
            norm_patch: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = hidden_size
        self.norm_outputs = norm_outputs
        self.patch_norm = norm_patch
        self.window_size = ensure_tuple_rep(window_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # region Build transformer layers
        self.use_v2 = use_v2
        self.layers = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicSwinLayer(
                dim=int(hidden_size * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )

            self.layers.append(layer)

            if self.use_v2:
                layer = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=hidden_size * 2 ** i_layer,
                    out_channels=hidden_size * 2 ** i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                self.layers.append(layer)
        # endregion

        self.pooling = TransformerPooling(pooling, dim=2)

        self.num_features = int(hidden_size * 2 ** self.num_layers)
        self.norm = norm_layer(self.num_features)

        self.project_outputs = (self.num_features != output_dimension) or (cut_init is not None)
        if self.project_outputs:
            self.linear = nn.Linear(self.num_features, output_dimension)
            if cut_init is not None:
                bound = 1.0 / math.sqrt(output_dimension) / cut_init
                nn.init.uniform_(self.linear.weight, -bound, bound)
        else:
            self.linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        batch_size, num_features, *_ = x.shape
        x = x.view(batch_size, num_features, -1)
        x = self.pooling(x)

        if self.norm_outputs:
            x = self.norm(x)

        if self.project_outputs:
            x = self.linear(x)

        return x
