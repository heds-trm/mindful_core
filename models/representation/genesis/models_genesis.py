import torch
from monai.networks.blocks import Convolution
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Union, Sequence, Any

from data.subset_id import SubsetID
from models.representation import AbstractRepresentationModel, RepresentationOutput
from models.loss_aggregator import LossAggregator
from models.representation.genesis.genesis_modules import GenesisDown, GenesisUp


class ModelsGenesis(AbstractRepresentationModel):
    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "models_genesis"

    @classmethod
    def module_aliases(cls) -> tuple[str, ...]:
        return ("genesis",)

    # endregion

    def __init__(self,
                 in_channels: int,
                 image_size: Sequence[int],
                 depth: int = 4,
                 stage_base_width: int = 8,
                 scaling: float | int = 2,
                 strides: Union[Sequence[int], int] = 2,
                 activation: str = "relu",
                 out_activation: str = "sigmoid",
                 normalization: str = None,
                 variance_lambda: float = 0.0,
                 covariance_lambda: float = 4e-2,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 callback_configs: dict[str, dict[str, Any]] = None,
                 *args,
                 **kwargs):
        output_dimension = self.compute_output_dimension(spatial_shape=image_size,
                                                         strides=strides,
                                                         depth=depth,
                                                         stage_base_width=stage_base_width,
                                                         scaling=scaling)
        super(ModelsGenesis, self).__init__(output_dimension=output_dimension,
                                            variance_lambda=variance_lambda,
                                            covariance_lambda=covariance_lambda,
                                            callback_configs=callback_configs,
                                            optimizer_config=optimizer_config,
                                            *args, **kwargs)
        self.in_channels = in_channels
        self.image_size = image_size
        self.depth = depth
        self.stage_base_width = stage_base_width
        self.scaling = scaling
        self.strides = strides
        self.activation = activation
        self.out_activation = out_activation
        self.normalization = normalization

        self.down_layers = []
        self.up_layers = []
        self.output_layer = None
        self.build_layers()

        self.train_sample_batch = None
        self.logged_train_sample_batch = False

    # region Setup
    def build_layers(self):
        stage_in_channels = self.in_channels
        down_layers = []
        intermediate_channels = []
        for i in range(1, self.depth + 1):
            stage_out_channels = self.stage_base_width * int(self.scaling ** i)
            stage_strides = 1 if i == self.depth else self.strides
            activation = self.activation if i < self.depth else "tanh"
            normalization = self.normalization if i < self.depth else None
            down_layer = GenesisDown(in_channels=stage_in_channels,
                                     out_channels=stage_out_channels,
                                     spatial_dims=self.spatial_dims,
                                     strides=stage_strides,
                                     activation=activation,
                                     normalization=normalization)
            down_layers.append(down_layer)
            self.add_module("down_layer_{}".format(i), down_layer)
            intermediate_channels.append(stage_out_channels)
            stage_in_channels = stage_out_channels
        self.down_layers = down_layers

        up_layers = []
        for i in range(self.depth - 1, 0, -1):
            stage_in_channels = intermediate_channels[i]
            stage_out_channels = intermediate_channels[i - 1]
            up_layer = GenesisUp(in_channels=stage_in_channels,
                                 out_channels=stage_out_channels,
                                 spatial_dims=self.spatial_dims,
                                 strides=self.strides,
                                 activation=self.activation,
                                 normalization=self.normalization)
            self.add_module("up_layer_{}".format(i), up_layer)
            up_layers.append(up_layer)
        self.up_layers = up_layers

        # noinspection PyTypeChecker
        self.output_layer = Convolution(spatial_dims=self.spatial_dims,
                                        in_channels=intermediate_channels[0],
                                        out_channels=self.in_channels,
                                        strides=1,
                                        act=self.out_activation,
                                        norm=None,
                                        dropout=None,
                                        padding="same")

    # endregion

    # region Forward
    def forward(self,
                inputs: torch.Tensor,
                *args,
                flatten: bool = True,
                **kwargs) -> RepresentationOutput:
        intermediate_outputs = inputs
        for down_layer in self.down_layers:
            intermediate_outputs, skip_outputs = down_layer(intermediate_outputs)
            del skip_outputs
        representations = intermediate_outputs

        if flatten:
            representations = representations.flatten(start_dim=1)

        return RepresentationOutput(representations, intermediate_outputs=None)

    def reconstruct(self, inputs, yield_representations=False):
        all_skip_outputs = []

        intermediate_outputs = inputs
        for down_layer in self.down_layers:
            intermediate_outputs, skip_outputs = down_layer(intermediate_outputs)
            all_skip_outputs.append(skip_outputs)
        representations = intermediate_outputs

        i = 0
        for up_layer, skip_outputs in zip(self.up_layers, reversed(all_skip_outputs[:-1])):
            i += 1
            intermediate_outputs = up_layer(intermediate_outputs, skip_outputs)

        outputs = self.output_layer(intermediate_outputs)

        if yield_representations:
            return outputs, representations
        else:
            return outputs

    # endregion

    # region Training
    def base_step(self, inputs, subset_id: SubsetID, *args, **kwargs) -> STEP_OUTPUT:
        self.loss_aggregator.clear()

        if self.train_sample_batch is None:
            self.train_sample_batch = inputs
        inputs, target = inputs
        outputs, representations = self.reconstruct(inputs, yield_representations=True)

        self.compute_base_loss(outputs, target)
        self.compute_variance_covariance_loss(representations)

        self.log_step_losses(subset_id)
        loss, logged_losses = self.loss_aggregator.get_logged_losses()
        step_outputs = {
            **logged_losses,
            "representations": representations.detach(),
        }
        self.step_outputs[subset_id].append(step_outputs)
        return loss

    def compute_base_loss(self, outputs: torch.Tensor, target: torch.Tensor) -> LossAggregator:
        base_loss = torch.mean(torch.pow(outputs - target, 2))
        self.loss_aggregator.add_loss(base_loss, "base_loss")
        return self.loss_aggregator

    # endregion

    # region Properties

    @property
    def spatial_dims(self) -> int:
        return len(self.image_size)

    @property
    def class_activation_target_layer(self) -> str:
        return ""

    # endregion

    @staticmethod
    def compute_output_dimension(spatial_shape: Union[int, Sequence[int]],
                                 strides: Union[int, Sequence[int]],
                                 depth: int,
                                 stage_base_width: int,
                                 scaling: float | int):
        if isinstance(strides, int):
            strides_reduction = [strides ** (depth - 1)]
        else:
            strides_reduction = [stride ** (depth - 1) for stride in strides]

        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]

        if len(spatial_shape) != len(strides_reduction):
            if (len(spatial_shape) != 1) and (len(strides_reduction) != 1):
                raise RuntimeError("Found strides and spatial_shape with different lengths.")
            if len(spatial_shape) == 1:
                spatial_shape = spatial_shape * len(strides_reduction)
            else:
                strides_reduction = strides_reduction * len(spatial_shape)

        output_dimension = stage_base_width * int(scaling ** depth)
        for spatial_size, stride_reduction in zip(spatial_shape, strides_reduction):
            output_dimension *= spatial_size // stride_reduction

        return output_dimension
