import torch
import torch.nn as nn
from monai.networks.nets import resnet, ResNet
from monai.networks.nets.densenet import DenseNet121
import copy
import os.path
from typing import Any, Callable, Sequence, Literal

from mindful_core.models.representation.encoders.vit_encoder import ViTEncoder, SwinEncoder
from mindful_core.models.representation.encoders.st_vit import STViT
from mindful_core.models.representation.encoders.vgg import VGGEncoder
from mindful_core.models.representation.encoders.multimodal.multimodal_encoder import MultiModalEncoder
from mindful_core.models.representation.encoders.misc import ScalarEncoder, CategoricalEncoder
from mindful_core.models.representation.encoders.multimodal import (
    FusionModule,

    ConcatFusion,
    StackFusion,

    MaxFusion,
    MultiplyFusion,
    AdditiveFusion,
    MeanFusion,

    AttentionAddFusion,
    MultiHeadAttentionFusion,
    FusionTransformer,
    LSTMFusion,

    HiNetFusion,
)
from mindful_core.models.classification.monai_classifier import MonaiClassifier, SUPPORTED_BACKBONES


# noinspection PyTypeChecker
def make_encoder(encoder_config: dict[str, Any]) -> nn.Module:
    encoder_config = copy.deepcopy(encoder_config)

    # region Extra config
    if "architecture" not in encoder_config:
        raise ValueError("Unknown architecture, key `architecture` is missing from encoder config: {}."
                         .format(encoder_config))
    architecture = encoder_config.pop("architecture").lower()
    checkpoint = encoder_config.pop("checkpoint") if "checkpoint" in encoder_config else None
    # endregion

    # region Architecture selection
    encoder_maker: Callable[[Any], nn.Module]
    if architecture in ENCODERS_REGISTER:
        encoder_maker = ENCODERS_REGISTER[architecture]
    
    elif "resnet" in architecture:
        if "resnet50" in architecture:
            encoder_config["layers"] = [3, 4, 6, 3]
            encoder_config["block"] = "ResNetBottleneck"
        elif "resnet34" in architecture:
            encoder_config["layers"] = [3, 4, 6, 3]
            encoder_config["block"] = "ResNetBlock"
        encoder_maker = make_resnet_encoder
    elif "densenet" in architecture:
        encoder_maker = make_densenet_encoder
    elif architecture in SUPPORTED_BACKBONES:
        encoder_maker = MonaiClassifier
        encoder_config["model_name"] = architecture
    else:
        raise RuntimeError("{} is currently an invalid encoder architecture.".format(architecture))

    encoder = encoder_maker(**encoder_config)
    # try:

    # except TypeError as original_error:
    #     raise TypeError("The configuration for encoder `{}` is either missing an argument or received an "
    #                     "unexpected one: {}.".format(architecture, original_error))
    # endregion

    # region Load checkpoint if provided
    if checkpoint and os.path.isfile(checkpoint):
        state_dict = torch.load(checkpoint)
        encoder.load_state_dict(state_dict)
    # endregion

    return encoder


# region Multimodal encoder
def make_encoders(encoders_config: dict[str, dict[str, Any] | str]) -> dict[str, nn.Module]:
    """
    Builds encoders for each (`modality`, `encoder_config`) pair in encoders_config.

    `encoder_config` can be a string referencing another modality. If so, it will reference the same encoder,
    thus sharing weights.
    """
    original_order = list(encoders_config.keys())

    encoders = {
        modality: make_encoder(encoder_config)
        for modality, encoder_config in encoders_config.items()
        if isinstance(encoder_config, dict)
    }
    
    reorder = False
    for modality, encoder_config in encoders_config.items():
        if isinstance(encoder_config, str):
            if encoder_config not in encoders_config:
                raise RuntimeError("Misconfiguration error: could not find encoder for modality `{}` in ({}).".
                                   format(encoder_config, list(encoders_config.keys())))
            encoders[modality] = encoders[encoder_config]
            reorder = True
        elif not isinstance(encoder_config, dict):
            raise RuntimeError("Misconfiguration error: Unknown identifier type `{}` for modality `{}`.".
                               format(type(encoder_config), modality))
        
    if reorder:
        encoders = {modality: encoders[modality] for modality in original_order}
        
    return encoders


# noinspection PyUnusedLocal
def make_multimodal_encoder(encoders_configs: dict[str, dict[str, Any]],
                            fusion_module_config: dict[str, Any],
                            output_module_config: dict[str, Any] = None,
                            **kwargs) -> MultiModalEncoder:
    modalities = list(encoders_configs.keys())

    encoders = list(make_encoders(encoders_configs).values())
    fusion_module = make_fusion_module(fusion_module_config)

    output_module: nn.Module | None
    if output_module_config is not None:
        output_module = make_encoder(output_module_config)
    else:
        output_module = None

    intermediate_sizes = []
    for base_modality in modalities:
        if isinstance(encoders_configs[base_modality], str):
            modality = encoders_configs[base_modality]
        else:
            modality = base_modality

        modality_config = encoders_configs[modality]
        if not isinstance(modality_config, dict):
            raise RuntimeError("Misconfiguration error for modality `{}`: Could not identify output dimension.".
                               format(base_modality))
        
        intermediate_sizes.append(modality_config["output_dimension"])

    return MultiModalEncoder(encoders=encoders, fusion_module=fusion_module, output_module=output_module,
                             modalities=modalities, intermediate_sizes=intermediate_sizes, **kwargs)


# region Fusion modules
FUSION_MODULES = {
    ConcatFusion: ["concatenate", "concat", "cat"],
    AdditiveFusion: ["add", "additive", "addition", "sum"],
    MeanFusion: ["mean", "average"],
    MaxFusion: ["max", "maximum"],
    MultiplyFusion: ["mul", "multiply", "product"],
    AttentionAddFusion: ["attention_add"],
    MultiHeadAttentionFusion: ["attention", "multihead_attention"],
    FusionTransformer: ["transformer"],
    LSTMFusion: ["lstm"],
    StackFusion: ["stack"],
    HiNetFusion: ["hinet", "hi-net"],
}


def make_fusion_module(config: dict[str, Any]) -> FusionModule:
    config = copy.deepcopy(config)
    # region Get fusion mode from config
    if "fusion_mode" not in config:
        raise ValueError("Unknown fusion module, key `fusion_mode` is missing from module config: {}.".format(config))
    fusion_mode = config.pop("fusion_mode").lower()
    # endregion

    # region Fetch the correct fusion module class
    module_class: FusionModule | None = None
    for available_module_class, aliases in FUSION_MODULES.items():
        if fusion_mode in aliases:
            # noinspection PyTypeChecker
            module_class = available_module_class
    # endregion

    # region Handle exceptions (unknown class, Hi-Net fusion, ...)
    if module_class is None:
        ValueError("Unknown fusion mode `{}`.".format(fusion_mode))
    elif module_class == HiNetFusion:
        submodules = [make_fusion_module(module_config) for module_config in config["modules"]]
        return HiNetFusion(submodules)
    # endregion

    return module_class(**config)


# endregion
# endregion

# noinspection PyUnusedLocal
def make_resnet_encoder(in_channels: int,
                        output_dimension: int,
                        layers: list[int],
                        feed_forward: bool,
                        spatial_dims: int = 3,
                        block: str = "ResNetBottleneck",
                        in_planes=None,
                        **kwargs
                        ) -> ResNet:
    if block == "ResNetBottleneck":
        block = resnet.ResNetBottleneck
    else:
        block = resnet.ResNetBlock

    if in_planes is None:
        in_planes = [
            output_dimension // 8,
            output_dimension // 4,
            output_dimension // 2,
            output_dimension
        ]
    elif not feed_forward:
        out_planes = in_planes[-1] * block.expansion
        if out_planes != output_dimension:
            raise ValueError("When not using the final feed-forward layer and providing `in_planes`,"
                             " the last value must match `output_dimension`, taking the block expansion into account"
                             " (output_dimension: {}, last in_plane: {} * expansion: {} = out_plane: {}."
                             .format(output_dimension, in_planes[-1], block.expansion, out_planes))

    encoder = ResNet(block=block,
                     layers=layers,
                     block_inplanes=in_planes,
                     spatial_dims=spatial_dims,
                     n_input_channels=in_channels,
                     conv1_t_size=7,
                     conv1_t_stride=1,
                     no_max_pool=False,
                     shortcut_type="B",
                     widen_factor=1.0,
                     num_classes=output_dimension,
                     feed_forward=feed_forward,
                     )

    return encoder


# noinspection PyUnusedLocal
def make_vit_encoder(image_size: int | tuple[int, int, int],
                     patch_size: int,
                     in_channels: int,
                     num_vit_layers: int,
                     num_vit_heads: int,
                     output_dimension: int,
                     spatial_dims: int,
                     hidden_size: int = None,
                     mlp_dim: int = None,
                     proj_type: str = "conv",
                     **kwargs
                     ) -> ViTEncoder:
    hidden_size = output_dimension if hidden_size is None else hidden_size
    mlp_dim = output_dimension * 2 if mlp_dim is None else mlp_dim
    encoder = ViTEncoder(in_channels=in_channels, img_size=image_size, patch_size=patch_size,
                         hidden_size=hidden_size, mlp_dim=mlp_dim, output_dimension=output_dimension,
                         num_layers=num_vit_layers, num_heads=num_vit_heads,
                         spatial_dims=spatial_dims, proj_type=proj_type, **kwargs)
    return encoder


def make_swin_encoder(window_size: int,
                      patch_size: int,
                      in_channels: int,
                      depths: Sequence[int],
                      num_heads: Sequence[int],
                      output_dimension: int,
                      spatial_dims: int,
                      pooling: Literal["cls", "first", "mean", "max", "mul", "prod", "none"],
                      hidden_size: int = None,
                      **kwargs
                      ) -> SwinEncoder:
    num_layers = len(depths)
    num_features = int(hidden_size * 2 ** (num_layers - 1))
    hidden_size = (output_dimension // num_features) if hidden_size is None else hidden_size

    encoder = SwinEncoder(in_channels=in_channels, window_size=window_size, patch_size=patch_size,
                          hidden_size=hidden_size, depths=depths, num_heads=num_heads,
                          pooling=pooling, output_dimension=output_dimension,
                          spatial_dims=spatial_dims,
                          **kwargs
                          )
    return encoder


def make_st_vit_encoder(image_size: int | tuple[int, int, int],
                        patch_size: int,
                        in_channels: int,
                        num_vit_layers: int,
                        num_vit_heads: int,
                        output_dimension: int,
                        st_kernel_size: int | tuple[int, int, int] = None,
                        spatial_dims: int = 3,
                        hidden_size: int = None,
                        mlp_dim: int = None,
                        proj_type: str = "conv",
                        **kwargs) -> STViT:
    hidden_size = output_dimension if hidden_size is None else hidden_size
    mlp_dim = output_dimension * 2 if mlp_dim is None else mlp_dim
    encoder = STViT(in_channels=in_channels, img_size=image_size, patch_size=patch_size,
                    st_kernel_size=st_kernel_size,
                    hidden_size=hidden_size, mlp_dim=mlp_dim, output_dimension=output_dimension,
                    num_layers=num_vit_layers, num_heads=num_vit_heads,
                    spatial_dims=spatial_dims, proj_type=proj_type, **kwargs)
    return encoder


def make_densenet_encoder(in_channels: int,
                          output_dimension: int,
                          spatial_dims: int = 3,
                          init_features: int = 64,
                          growth_rate: int = 32,
                          block_config: Sequence[int] = (6, 12, 24, 16),
                          pretrained: bool = False,
                          ) -> DenseNet121:
    return DenseNet121(spatial_dims=spatial_dims,
                       in_channels=in_channels,
                       out_channels=output_dimension,
                       init_features=init_features,
                       growth_rate=growth_rate,
                       block_config=block_config,
                       pretrained=pretrained,
                       progress=False)


def make_vgg_encoder(in_channels: int,
                     depth: tuple[int, int, int, int, int] | int = (1, 1, 1, 1, 1),
                     width: tuple[int, int, int, int, int] | int = (64, 128, 256, 512, 512),
                     kernel_size: int = 7,
                     spatial_dims: int = 3,
                     use_batch_norm: bool = False,
                     ) -> VGGEncoder:
    if isinstance(depth, int):
        depth = tuple([depth] * 5)

    if isinstance(width, int):
        width = (width, width * 2, width * 4, width * 8, width * 8)

    return VGGEncoder(in_channels=in_channels,
                      depth=depth,
                      width=width,
                      kernel_size=kernel_size,
                      spatial_dims=spatial_dims,
                      use_batch_norm=use_batch_norm)


def make_scalar_encoder(input_size: int,
                        hidden_sizes: list[int] | int,
                        output_dimension: int, ) -> ScalarEncoder:
    return ScalarEncoder(input_size=input_size, hidden_sizes=hidden_sizes, output_dimension=output_dimension)


def make_categorical_encoder(categories_sizes: list[int],
                             categories_hidden_sizes: list[int] | int,
                             linear_hidden_sizes: list[int] | int,
                             output_dimension: int,
                             add_missing_token: bool = True) -> CategoricalEncoder:
    return CategoricalEncoder(categories_sizes=categories_sizes,
                              categories_hidden_sizes=categories_hidden_sizes,
                              linear_hidden_sizes=linear_hidden_sizes,
                              output_dimension=output_dimension,
                              add_missing_token=add_missing_token)

ENCODERS_REGISTER = {
    "st_vit": make_st_vit_encoder,
    "vit": make_vit_encoder,
    "swin": make_swin_encoder,
    "multimodal": make_multimodal_encoder,
    "scalar": make_scalar_encoder,
    "categorical": make_categorical_encoder,
    "vgg": make_vgg_encoder
}
