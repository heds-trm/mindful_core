import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any

from mindful_core.models.representation.abstract_representation_model import AbstractRepresentationModel, RepresentationOutput
from mindful_core.models.representation.encoders.factory import make_encoder
from mindful_core.data.subset_id import SubsetID


class EncoderRepresentationModel(AbstractRepresentationModel):
    @classmethod
    def module_identifier(cls) -> str:
        return "encoder"

    @classmethod
    def module_aliases(cls) -> tuple[str, ...]:
        return ()

    def __init__(self,
                 encoder_config: dict[str, Any],
                 variance_lambda: float = 0.0,
                 covariance_lambda: float = 4e-2,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 callback_configs: dict[str, dict[str, Any]] = None,
                 *args,
                 **kwargs
                 ):
        if "output_dimension" in encoder_config:
            output_dimension = encoder_config["output_dimension"]
        elif "output_dimension" in kwargs:
            output_dimension = kwargs["output_dimension"]
        else:
            raise ValueError("`output_dimension` is missing from encoder_config. Got {}.".format(encoder_config))

        if "output_dimension" in kwargs:
            kwargs.pop("output_dimension")

        if "image_size" in kwargs:
            encoder_config["image_size"] = kwargs.pop("image_size")

        super(EncoderRepresentationModel, self).__init__(output_dimension=output_dimension,
                                                         covariance_lambda=covariance_lambda,
                                                         variance_lambda=variance_lambda,
                                                         callback_configs=callback_configs,
                                                         optimizer_config=optimizer_config,
                                                         *args, **kwargs)
        self.encoder_config = encoder_config
        self.encoder = make_encoder(encoder_config)

    def forward(self,
                inputs: torch.Tensor,
                *args,
                flatten: bool = True,
                output_intermediates: bool = False,
                **kwargs
                ) -> RepresentationOutput:
        return self._get_model_representations(self.encoder, inputs, flatten, output_intermediates)

    def base_step(self, inputs, subset_id: SubsetID, *args, **kwargs) -> STEP_OUTPUT:
        raise NotImplementedError("EncoderRepresentationModel is not trainable as-is. "
                                  "`base_step` must be implemented in subclasses.")

    @property
    def spatial_dims(self):
        if "spatial_dims" not in self.encoder_config:
            return 3
        else:
            return self.encoder_config["spatial_dims"]
