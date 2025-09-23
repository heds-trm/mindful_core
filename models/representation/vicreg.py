from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from typing import Any

from data.subset_id import SubsetID
from models.representation import EncoderRepresentationModel, RepresentationOutput
from models.loss_aggregator import LossAggregator


class VICReg(EncoderRepresentationModel):
    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "vicreg"

    @classmethod
    def multiview(cls) -> bool | int:
        return 2

    # endregion

    def __init__(self,
                 encoder_config: dict[str, Any],
                 variance_lambda: float = 1.0,
                 covariance_lambda: float = 4e-2,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 callback_configs: dict[str, dict[str, Any]] = None,
                 *args, **kwargs):
        super(VICReg, self).__init__(encoder_config=encoder_config,
                                     variance_lambda=variance_lambda,
                                     covariance_lambda=covariance_lambda,
                                     callback_configs=callback_configs,
                                     optimizer_config=optimizer_config,
                                     *args,
                                     **kwargs)
        self.save_hyperparameters()

        self.expander = nn.Sequential(
            nn.Linear(self.output_dimension, self.output_dimension * 2),
            nn.BatchNorm1d(self.output_dimension * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dimension * 2, self.output_dimension * 4, bias=False)
        )

    def base_step(self, inputs, subset_id: SubsetID, *args, **kwargs) -> STEP_OUTPUT:
        self.loss_aggregator.clear()

        inputs_view_1, inputs_view_2 = inputs

        outputs_1: RepresentationOutput = self(inputs_view_1)
        outputs_2: RepresentationOutput = self(inputs_view_2)
        representations_1 = outputs_1.representations
        representations_2 = outputs_2.representations
        representations = (representations_1, representations_2)

        self.compute_base_loss(representations_1, representations_2)
        self.compute_variance_covariance_loss(representations)
        self.log_step_losses(subset_id)

        self.log_step_losses(subset_id)
        loss, logged_losses = self.loss_aggregator.get_logged_losses()
        step_outputs = {
            **logged_losses,
            "representations": representations_1.detach(),
        }
        self.step_outputs[subset_id].append(step_outputs)
        return loss

    def compute_base_loss(self,
                          representations_1: torch.Tensor,
                          representations_2: torch.Tensor
                          ) -> LossAggregator:
        expanded_1 = self.expander(representations_1)
        expanded_2 = self.expander(representations_2)

        base_loss = torch.pow(expanded_1 - expanded_2, 2.0).mean()
        self.loss_aggregator.add_loss(base_loss, "base_loss")

        return self.loss_aggregator
