import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: list[int],
                 width: list[int],
                 kernel_size: int | list[int] | list[list[int]],
                 stride: int | list[int] | list[list[int]],
                 padding: int | list[int] | list[list[int]],
                 output_padding: int | list[int] | list[list[int]],
                 stem_size: int,
                 spatial_dims: int = 3,
                 use_batch_norm: bool = False,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.width = width
        self.kernel_size = self._normalize_depth(kernel_size)
        self.stride = self._normalize_depth(stride)
        self.padding = self._normalize_depth(padding)
        self.output_padding = self._normalize_depth(output_padding)
        self.stem_size = stem_size
        self.spatial_dims = spatial_dims
        self.use_batch_norm = use_batch_norm

        blocks = [self._make_block(depth[i],
                                   self._in_features(i),
                                   self.width[i],
                                   self.kernel_size[i],
                                   self.padding[i],
                                   self.stride[i],
                                   self.output_padding[i]
                                   )
                  for i in range(len(depth))]

        self.blocks = blocks
        self.stem = self._make_stem()

        self.decode = nn.Sequential(*blocks, self.stem)

    # noinspection PyUnusedLocal
    def forward(self, inputs: torch.Tensor, *args, **kwargs):
        if len(inputs.shape) != (self.spatial_dims + 2):
            if len(inputs.shape) != 2:
                raise RuntimeError("Could not broadcast rank `{}` to `{}`".
                                   format(len(inputs.shape), self.spatial_dims + 2))
            batch_size, features = inputs.shape
            inputs = inputs.view(batch_size, features, *([1] * self.spatial_dims))
        return self.decode(inputs)

    def _normalize_depth(self, value: int | list[int] | list[list[int]]) -> list[int] | list[list[int]]:
        if isinstance(value, int) or (len(value) != len(self.depth)):
            value = [value] * len(self.depth)
        return value

    def _normalize_spatial_dims(self, value: int | list[int]) -> list[int]:
        if isinstance(value, int):
            value = [value] * self.spatial_dims
        return value

    def _make_block(self,
                    depth: int,
                    in_channels: int,
                    out_channels: int,
                    kernel_size: list[int] | int,
                    padding: list[int] | int,
                    stride: list[int] | int,
                    output_padding: list[int] | int,
                    ) -> nn.Sequential:
        kernel_size = self._normalize_spatial_dims(kernel_size)
        padding = self._normalize_spatial_dims(padding)
        stride = self._normalize_spatial_dims(stride)
        output_padding = self._normalize_spatial_dims(output_padding)

        layers = []
        for _ in range(depth):
            # region Conv
            conv_params = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "output_padding": output_padding,
            }
            if self.spatial_dims == 1:
                layer = nn.ConvTranspose1d(**conv_params)
            elif self.spatial_dims == 2:
                layer = nn.ConvTranspose2d(**conv_params)
            else:
                layer = nn.ConvTranspose3d(**conv_params)

            layers.append(layer)
            in_channels = out_channels
            # endregion

            # region Batch normalization
            if self.use_batch_norm:
                if self.spatial_dims == 1:
                    layer = nn.BatchNorm1d(out_channels)
                elif self.spatial_dims == 2:
                    layer = nn.BatchNorm2d(out_channels)
                else:
                    layer = nn.BatchNorm3d(out_channels)

                layers.append(layer)
            # endregion

            # region ReLU
            layers.append(nn.ReLU())

            stride = [1] * self.spatial_dims
            padding = [x // 2 for x in kernel_size]
            output_padding = [0] * self.spatial_dims

            # endregion

        return nn.Sequential(*layers)

    def _in_features(self, i: int) -> int:
        if i == 0:
            return self.in_channels
        else:
            return self.width[i - 1]

    def _make_stem(self) -> nn.Conv1d | nn.Conv2d | nn.Conv3d:
        conv_params = {
            "in_channels": self.width[-1],
            "out_channels": self.out_channels,
            "kernel_size": self.stem_size,
            "stride": 1,
            "padding": self.stem_size // 2,
        }

        if self.spatial_dims == 1:
            layer = nn.Conv1d(**conv_params)
        elif self.spatial_dims == 2:
            layer = nn.Conv2d(**conv_params)
        else:
            layer = nn.Conv3d(**conv_params)

        return layer
