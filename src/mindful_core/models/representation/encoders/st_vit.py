import torch
import torch.nn as nn
from torch.nn.functional import grid_sample, affine_grid
from monai.utils import ensure_tuple_rep
import numpy as np
from typing import Sequence, Literal

from mindful_core.models.representation.encoders.vit_encoder import ViTEncoder


class STViT(ViTEncoder):
    def __init__(self,
                 in_channels: int,
                 img_size: Sequence[int] | int,
                 patch_size: Sequence[int] | int,
                 st_kernel_size: Sequence[int] | int = None,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 output_dimension: int = 512,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 proj_type: str = "conv",
                 dropout_rate: float = 0.0,
                 norm_outputs: bool = False,
                 spatial_dims: int = 3,
                 pooling: int | Literal["cls", "first", "mean", "max", "mul", "prod", "none"] | None = "first",
                 ):
        if spatial_dims != 3:
            raise NotImplementedError(spatial_dims)

        if st_kernel_size is None:
            st_kernel_size = patch_size
        super(STViT, self).__init__(in_channels=in_channels,
                                    img_size=img_size,
                                    patch_size=patch_size,
                                    hidden_size=hidden_size,
                                    mlp_dim=mlp_dim,
                                    output_dimension=output_dimension,
                                    num_layers=num_layers,
                                    num_heads=num_heads,
                                    proj_type=proj_type,
                                    dropout_rate=dropout_rate,
                                    norm_outputs=norm_outputs,
                                    spatial_dims=spatial_dims,
                                    pooling=pooling,
                                    )

        st_kernel_size = ensure_tuple_rep(st_kernel_size, dim=self.spatial_dims)
        theta_size = 6 if spatial_dims == 2 else 12

        # noinspection PyTypeChecker
        self.st_pooling = nn.AvgPool3d(kernel_size=st_kernel_size)
        self.st_input_dim = np.prod([image_dim // patch_dim for image_dim, patch_dim
                                     in zip(self.img_size, self.patch_size)]) * in_channels
        self.st_expand = nn.Linear(self.st_input_dim, self.st_input_dim * 2)
        self.st_proj = nn.Linear(self.st_input_dim * 2, theta_size)
        self.st_flatten = nn.Flatten()
        self.st_relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor, *args, **kwargs):
        st_image = inputs
        st_image = self.st_pooling(st_image)
        st_image = self.st_flatten(st_image)
        st_image = self.st_expand(st_image)
        st_image = self.st_relu(st_image)
        theta: torch.Tensor = self.st_proj(st_image)
        if self.spatial_dims == 2:
            theta = theta.view(-1, 2, 3)
        else:
            theta = theta.view(-1, 3, 4)
        grid = affine_grid(theta, list(inputs.shape))
        inputs = grid_sample(inputs, grid)

        outputs = super().forward(inputs, *args, **kwargs)
        return outputs
