import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
import numpy as np
from typing import Union, Sequence


class GenesisBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: int = 2,
                 progressive_channels: bool = True,
                 spatial_dims: int = 2,
                 kernel_size: Union[Sequence[int], int] = 3,
                 activation: str = "relu",
                 normalization: str = "batch",
                 dropout: float = None,
                 ):
        super(GenesisBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.progressive_channels = progressive_channels
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.activation = activation
        self.normalization = normalization
        self.dropout = dropout

        stage_in_channels = in_channels
        conv_layers = []
        for stage_out_channels in self.stage_out_channels_generator():
            # noinspection PyTypeChecker
            conv_layer = Convolution(spatial_dims=self.spatial_dims,
                                     in_channels=stage_in_channels,
                                     out_channels=stage_out_channels,
                                     strides=1,
                                     kernel_size=self.kernel_size,
                                     act=self.activation,
                                     norm=self.normalization,
                                     dropout=self.dropout,
                                     padding="same",
                                     )
            conv_layers.append(conv_layer)
            stage_in_channels = stage_out_channels
        self.conv_layers = nn.Sequential(*conv_layers)

    def stage_out_channels_generator(self):
        if not self.progressive_channels:
            for _ in range(self.depth):
                yield self.out_channels
        else:
            in_out_ratio = self.out_channels / self.in_channels
            stage_in_out_ratio = np.power(in_out_ratio, 1 / self.depth)
            for i in range(1, self.depth + 1):
                if i == self.depth:
                    yield self.out_channels
                else:
                    yield int(round(self.in_channels * stage_in_out_ratio ** i))

    def forward(self, inputs):
        return self.conv_layers(inputs)


class GenesisDown(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: int = 2,
                 progressive_channels: bool = True,
                 spatial_dims: int = 2,
                 kernel_size: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 2,
                 activation: str = "relu",
                 normalization: str = "batch",
                 dropout: float = None,
                 ):
        super(GenesisDown, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.progressive_channels = progressive_channels
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.normalization = normalization
        self.dropout = dropout

        if isinstance(strides, int):
            use_max_pool = strides != 1
        else:
            use_max_pool = all([stride != 1 for stride in strides])
        self.use_max_pool = use_max_pool

        self.block = GenesisBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  depth=depth,
                                  progressive_channels=progressive_channels,
                                  spatial_dims=spatial_dims,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  normalization=normalization,
                                  dropout=dropout)

        if self.use_max_pool:
            self.max_pool = self.get_max_pool_layer()
        else:
            self.max_pool = None

    def get_max_pool_layer(self):
        if self.spatial_dims == 1:
            return nn.MaxPool1d(self.strides)
        elif self.spatial_dims == 2:
            return nn.MaxPool2d(self.strides)
        elif self.spatial_dims == 3:
            return nn.MaxPool3d(self.strides)
        else:
            raise ValueError("Spatial dims must be between 1 and 3 (included).")

    def forward(self, inputs):
        intermediate_output = self.block(inputs)
        if self.use_max_pool:
            outputs = self.max_pool(intermediate_output)
        else:
            outputs = intermediate_output
        return outputs, intermediate_output


class GenesisUp(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 concat_skip: bool = True,
                 depth: int = 2,
                 progressive_channels: bool = False,
                 spatial_dims: int = 2,
                 kernel_size: Union[Sequence[int], int] = 3,
                 strides: Union[Sequence[int], int] = 2,
                 activation: str = "relu",
                 normalization: str = "batch",
                 dropout: float = None,
                 ):
        super(GenesisUp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat_skip = concat_skip
        self.depth = depth
        self.progressive_channels = progressive_channels
        self.spatial_dims = spatial_dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.normalization = normalization
        self.dropout = dropout

        self.up_conv_layer = Convolution(spatial_dims=self.spatial_dims,
                                         in_channels=in_channels,
                                         out_channels=out_channels,
                                         strides=strides,
                                         kernel_size=kernel_size,
                                         is_transposed=True,
                                         norm=None,
                                         dropout=None,
                                         padding=1)

        block_in_channels = out_channels * 2 if self.concat_skip else out_channels
        self.block = GenesisBlock(in_channels=block_in_channels,
                                  out_channels=out_channels,
                                  depth=depth,
                                  progressive_channels=progressive_channels,
                                  spatial_dims=spatial_dims,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  normalization=normalization,
                                  dropout=dropout)

    def forward(self, inputs, skip_inputs):
        intermediate_outputs = self.up_conv_layer(inputs)
        if self.concat_skip:
            intermediate_outputs = torch.cat([intermediate_outputs, skip_inputs], dim=1)
        else:
            intermediate_outputs = intermediate_outputs + skip_inputs
        outputs = self.block(intermediate_outputs)
        return outputs
