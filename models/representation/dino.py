import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from copy import deepcopy
from typing import Any, Iterator

from models.representation import EncoderRepresentationModel, RepresentationOutput
from data.subset_id import SubsetID
from utils.tensor_utils import lerp


class DINO(EncoderRepresentationModel):
    # region Class/Subclass methods
    @classmethod
    def module_identifier(cls) -> str:
        return "dino"

    @classmethod
    def multiview(cls) -> bool | int:
        return 2

    # endregion

    def __init__(self,
                 encoder_config: dict[str, Any],
                 student_temperature: float = 0.1,
                 teacher_temperature: float = 0.04,
                 teacher_momentum: float = 0.996,
                 teacher_center_momentum: float = 0.9,
                 variance_lambda: float = 0.0,
                 covariance_lambda: float = 4e-2,
                 optimizer_config: dict[str, dict[str, Any]] = None,
                 callback_configs: dict[str, dict[str, Any]] = None,
                 *args,
                 **kwargs
                 ):
        super(DINO, self).__init__(encoder_config=encoder_config,
                                   variance_lambda=variance_lambda,
                                   covariance_lambda=covariance_lambda,
                                   callback_configs=callback_configs,
                                   optimizer_config=optimizer_config,
                                   *args, **kwargs)
        # self.student_temperature = student_temperature
        # self.teacher_temperature = teacher_temperature
        self.teacher_momentum = teacher_momentum
        # self.teacher_center_momentum = teacher_center_momentum

        self.save_hyperparameters()

        self.student = deepcopy(self.teacher)
        self.load_and_freeze_teacher()

        self.loss_module = DINOLoss(self.output_dimension,
                                    student_temperature=student_temperature,
                                    teacher_temperature=teacher_temperature,
                                    teacher_center_momentum=teacher_center_momentum)

    # region Setup
    def load_and_freeze_teacher(self):
        self.teacher.load_state_dict(self.student.state_dict())
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

    @property
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.student.parameters()

    # endregion

    # region Forward
    def student_forward(self, inputs: torch.Tensor, flatten: bool = True) -> RepresentationOutput:
        return self._get_model_representations(self.student, inputs, flatten)

    def teacher_forward(self, inputs: torch.Tensor, flatten: bool = True) -> RepresentationOutput:
        return self._get_model_representations(self.teacher, inputs, flatten)

    # endregion

    # region Training
    def base_step(self, inputs, subset_id: SubsetID, *args, **kwargs) -> STEP_OUTPUT:
        self.loss_aggregator.clear()

        inputs_view_1, inputs_view_2 = inputs
        student_outputs = (self.student_forward(inputs_view_1).representations,
                           self.student_forward(inputs_view_2).representations)
        teacher_outputs = (self.teacher_forward(inputs_view_1).representations,
                           self.teacher_forward(inputs_view_2).representations)

        self.compute_base_loss(student_outputs, teacher_outputs)
        self.compute_variance_covariance_loss(student_outputs)

        self.log_step_losses(subset_id)
        loss, logged_losses = self.loss_aggregator.get_logged_losses()
        step_outputs = {
            **logged_losses,
            "representations": teacher_outputs[0].detach(),
        }
        self.step_outputs[subset_id].append(step_outputs)
        return loss

    def compute_base_loss(self,
                          student_outputs: tuple[torch.Tensor, torch.Tensor],
                          teacher_outputs: tuple[torch.Tensor, torch.Tensor]):
        base_loss = self.loss_module(student_outputs, teacher_outputs)
        self.loss_aggregator.add_loss(base_loss, "base_loss")
        return self.loss_aggregator

    def on_train_batch_end(self,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int
                           ) -> None:
        super(DINO, self).on_train_batch_end(outputs, batch, batch_idx)
        self.update_teacher_parameters()

    def update_teacher_parameters(self):
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
                student_param = student_param.detach()
                teacher_param.copy_(lerp(student_param, teacher_param, self.teacher_momentum))

    # endregion

    @property
    def teacher(self):
        return self.encoder


class DINOLoss(nn.Module):
    teacher_center: torch.Tensor

    def __init__(self,
                 output_dimension: int,
                 student_temperature: float = 0.1,
                 teacher_temperature: float = 0.04,
                 teacher_center_momentum: float = 0.9
                 ) -> None:
        super(DINOLoss, self).__init__()
        self.output_dimension = output_dimension
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.teacher_center_momentum = teacher_center_momentum

        self.register_buffer(name="teacher_center", tensor=torch.zeros(self.output_dimension))

    def forward(self,
                student_outputs: tuple[torch.Tensor, torch.Tensor],
                teacher_outputs: tuple[torch.Tensor, torch.Tensor]
                ) -> torch.Tensor:
        student_output_1, student_output_2 = student_outputs
        teacher_output_1, teacher_output_2 = teacher_outputs

        loss = (self.get_view_loss(student_output_1, teacher_output_2) +
                self.get_view_loss(student_output_2, teacher_output_1)) * 0.5

        if torch.is_grad_enabled():
            self.update_teacher_center(teacher_outputs)

        return loss

    def get_view_loss(self,
                      student_output: torch.Tensor,
                      teacher_output: torch.Tensor
                      ) -> torch.Tensor:
        log_student_output = torch.log_softmax(student_output / self.student_temperature, dim=-1)
        teacher_output = teacher_output.detach() - torch.unsqueeze(self.teacher_center, dim=0)
        teacher_output = torch.softmax(teacher_output / self.teacher_temperature, dim=-1)
        return - (teacher_output * log_student_output).sum(dim=-1).mean()

    def update_teacher_center(self, teacher_outputs: tuple[torch.Tensor, ...]):
        teacher_outputs = torch.cat(teacher_outputs, dim=0)
        batch_center = torch.mean(teacher_outputs, dim=0)
        self.teacher_center = lerp(batch_center, self.teacher_center, self.teacher_center_momentum)
