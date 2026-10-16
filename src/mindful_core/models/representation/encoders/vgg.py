import torch.nn as nn


class VGGEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 depth: tuple[int, int, int, int, int],
                 width: tuple[int, int, int, int, int],
                 kernel_size: int,
                 spatial_dims: int = 3,
                 use_batch_norm: bool = False,
                 ):
        super().__init__()
        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)

        self.in_channels = in_channels
        self.depth = depth
        self.width = width
        self.kernel_size = kernel_size
        self.spatial_dims = spatial_dims
        self.use_batch_norm = use_batch_norm

        blocks = [self._make_block(depth[i],
                                   self._in_features(i),
                                   self.width[i])
                  for i in range(len(depth))]

        self.blocks = blocks
        self.encode = nn.Sequential(*blocks)

    # noinspection PyUnusedLocal
    def forward(self, inputs, *args, **kwargs):
        return self.encode(inputs)

    def _make_block(self, depth: int, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = []
        for _ in range(depth):
            # region Conv
            if isinstance(self.kernel_size, int):
                padding = self.kernel_size // 2
            else:
                padding = tuple([k // 2 for k in self.kernel_size])
            conv_params = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": self.kernel_size,
                "stride": 1,
                "padding": padding,
            }
            if self.spatial_dims == 1:
                layer = nn.Conv1d(**conv_params)
            elif self.spatial_dims == 2:
                layer = nn.Conv2d(**conv_params)
            else:
                layer = nn.Conv3d(**conv_params)

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

            # endregion

        # region Max pooling
        if self.spatial_dims == 1:
            layer = nn.MaxPool1d(kernel_size=2, stride=2)
        elif self.spatial_dims == 2:
            layer = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            layer = nn.MaxPool3d(kernel_size=2, stride=2)
        # endregion

        layers.append(layer)

        return nn.Sequential(*layers)

    def _in_features(self, i: int) -> int:
        if i == 0:
            return self.in_channels
        else:
            return self.width[i - 1]
