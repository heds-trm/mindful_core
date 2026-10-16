import torch
import torch.nn as nn
import copy
import os
from typing import Any

from mindful_core.models.representation.encoders.misc import ScalarEncoder
from mindful_core.models.representation.decoders.deconv import ConvDecoder


def make_decoder(decoder_config: dict[str, Any]) -> nn.Module:
    decoder_config = copy.deepcopy(decoder_config)

    # region Extra config
    if "architecture" not in decoder_config:
        raise ValueError("Unknown architecture, key `architecture` is missing from encoder config: {}."
                         .format(decoder_config))
    architecture = decoder_config.pop("architecture").lower()
    checkpoint = decoder_config.pop("checkpoint") if "checkpoint" in decoder_config else None
    # endregion

    if architecture == "resnet":
        raise NotImplementedError
    elif architecture == "st_vit":
        raise NotImplementedError
    elif architecture == "vit":
        raise NotImplementedError
    elif architecture == "multimodal":
        raise NotImplementedError
    elif architecture in ["conv", "deconv"]:
        decoder_maker = make_conv_decoder
    elif architecture in ["scalar", "scalars"]:
        decoder_maker = make_scalar_decoder
    elif architecture == "categorical":
        raise NotImplementedError
    else:
        raise RuntimeError("{} is currently an invalid decoder architecture.".format(architecture))

    decoder = decoder_maker(**decoder_config)

    # region Load checkpoint if provided
    if checkpoint and os.path.isfile(checkpoint):
        state_dict = torch.load(checkpoint)
        decoder.load_state_dict(state_dict)
    # endregion

    return decoder


def make_scalar_decoder(input_size: int,
                        hidden_sizes: list[int] | int,
                        output_dimension: int) -> ScalarEncoder:
    """
    Same as encoders.factory.make_scalar_decoder() as ScalarEncoder can be used as a decoder.
    :param input_size:
    :param hidden_sizes:
    :param output_dimension:
    :return:
    """
    return ScalarEncoder(input_size=input_size, hidden_sizes=hidden_sizes, output_dimension=output_dimension)


def make_conv_decoder(in_channels: int,
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
                      ) -> ConvDecoder:
    return ConvDecoder(in_channels=in_channels,
                       out_channels=out_channels,
                       depth=depth,
                       width=width,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding,
                       output_padding=output_padding,
                       stem_size=stem_size,
                       spatial_dims=spatial_dims,
                       use_batch_norm=use_batch_norm
                       )

def make_decoders(decoders_config: dict[str, dict[str, Any] | str]) -> dict[str, nn.Module]:
    """
    Builds decoders for each (`modality`, `encoder_config`) pair in encoders_config.

    `decoder_config` can be a string referencing another modality. If so, it will reference the same decoder,
    thus sharing weights.
    """
    original_order = list(decoders_config.keys())

    decoders = {
        modality: make_decoder(decoder_config)
        for modality, decoder_config in decoders_config.items()
        if isinstance(decoder_config, dict)
    }
    
    reorder = False
    for modality, decoder_config in decoders_config.items():
        if isinstance(decoder_config, str):
            if decoder_config not in decoders_config:
                raise RuntimeError("Misconfiguration error: could not find decoder for modality `{}` in ({}).".
                                   format(decoder_config, list(decoders_config.keys())))
            decoders[modality] = decoders[decoder_config]
            reorder = True
        elif not isinstance(decoder_config, dict):
            raise RuntimeError("Misconfiguration error: Unknown identifier type `{}` for modality `{}`.".
                               format(type(decoder_config), modality))
        
    if reorder:
        decoders = {modality: decoders[modality] for modality in original_order}
        
    return decoders

