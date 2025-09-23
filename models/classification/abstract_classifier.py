import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import ROC, PrecisionRecallCurve, PearsonCorrCoef
from torch.nn.functional import binary_cross_entropy_with_logits, log_softmax, nll_loss
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from abc import abstractmethod
from typing import Union, Optional, Any, Type

from models.model_output import ClassifierOutput, PrototypeOutput
from models.module import MindfulModule, LossAggregator
from models.classification.prototype_layer import PrototypeLayer
from analysis.statistics.classification_summary import (
    ClassificationSummary,
    ClassificationMetric,

    AUROC,
    AboveThreshold,
    EER,
    EERThreshold,
    AccuracyAtEER,
    SensitivityAtEER,
    SpecificityAtEER,

    make_metrics_for_sensitivity_threshold,
    SensitivityThreshold,
    AccuracyAtSensitivity,
    SpecificityAtSensitivity,
    SensitivityAtSensitivity,

    ConfidenceThreshold,
    ConfidencePositivityCorrelation
)
from data.subset_id import SubsetID
from utils.visualization import plot_line2d_to_array
from utils.tensor_utils import lerp, get_gradients, set_require_grads


class AbstractClassifier(MindfulModule):
    """
        todo: write documentation
    """

    # region Class/Subclass methods
    @classmethod
    @abstractmethod
    def module_identifier(cls) -> str:
        raise NotImplementedError("Method `module_identifier` must be implemented in subclasses")

    @classmethod
    def get_registered_classifiers(cls) -> dict[str, Type["AbstractClassifier"]]:
        return {key: value for key, value in cls._module_register if issubclass(value, AbstractClassifier)}

    @classmethod
    def base_log_monitors(cls) -> list[str]:
        return ["loss", "auroc"]

    # endregion
    def __init__(self,
                 class_count: int,
                 optimizer_config: dict[str, dict[str, Any]] | None = None,
                 label_smoothing=0.0,

                 confidence_lambda=1e-1,
                 confidence_corr_lambda=1e-1,
                 confidence_budget=None,  # good default values are in [0.1, 1.0]

                 hidden_size: int | None = None,
                 prototype_model_config: dict | None = None,
                 prototype_lambda=1e-1,

                 positive_class=-1,
                 use_focal_loss=False,
                 **kwargs):
        super(AbstractClassifier, self).__init__(optimizer_config=optimizer_config)

        if class_count <= 0:
            raise ValueError("`class_count` must be strictly positive (got {})".format(class_count))

        self.class_count = class_count
        self.label_smoothing = label_smoothing
        self.positive_class = positive_class
        self.use_focal_loss = use_focal_loss

        self.base_loss_lambda = 1.0

        # region Confidence
        if confidence_budget is None:
            self.confidence_lambda = confidence_lambda
            self.confidence_budget = None
        else:
            confidence_lambda = torch.as_tensor(confidence_lambda, dtype=torch.float32)
            self.confidence_lambda = nn.Parameter(confidence_lambda, requires_grad=False)
            self.confidence_budget = confidence_budget

        self.confidence_corr_lambda = confidence_corr_lambda
        self.disable_confidence = False
        # endregion

        self.hidden_size = hidden_size

        # region Prototype model/layer
        self.prototype_model_config = prototype_model_config
        if prototype_model_config is not None:
            if "input_dimension" not in prototype_model_config:
                prototype_model_config["input_dimension"] = hidden_size
            if "class_count" not in prototype_model_config:
                prototype_model_config["class_count"] = class_count

            self.prototype_model = PrototypeLayer(**prototype_model_config)
        else:
            self.prototype_model = None
        self.prototype_lambda = prototype_lambda
        # endregion

        # region Classification summary
        metrics = [AUROC, AboveThreshold, EER,
                   AccuracyAtEER, SensitivityAtEER, SpecificityAtEER,
                   ConfidencePositivityCorrelation]

        sensitivity_metrics = make_metrics_for_sensitivity_threshold(base_metrics=[
            SensitivityThreshold,
            AccuracyAtSensitivity,
            SensitivityAtSensitivity,
            SpecificityAtSensitivity
        ],
            relative_threshold=.90)
        # noinspection PyTypeChecker
        self.sensitivity_threshold_class: type[SensitivityThreshold] = sensitivity_metrics.pop(0)
        metrics += sensitivity_metrics

        self.classification_summary = ClassificationSummary(metrics=metrics,
                                                            num_thresholds=1000,
                                                            confidence_threshold=None,
                                                            positive_class=positive_class)
        # endregion

    # region Forward
    def forward(self, inputs, *args, **kwargs) -> ClassifierOutput:
        raise NotImplementedError

    # endregion

    # region Training
    def base_step(self, batch, subset_id: SubsetID, **model_kwargs) -> STEP_OUTPUT:
        self.loss_aggregator.clear()
        inputs, labels = self.unpack_batch(batch)

        outputs: ClassifierOutput = self(inputs, **model_kwargs)
        if not isinstance(outputs, ClassifierOutput):
            raise RuntimeError("Output of subclasses of `AbstractClassifier` must be instances of `ClassifierOutput`.")

        self.compute_output_losses(outputs, labels, subset_id)

        self.log_step_losses(subset_id)
        loss, logged_losses = self.loss_aggregator.get_logged_losses()
        step_outputs = {
            **logged_losses,
            "labels": labels,
            **outputs.detached_outputs(),
        }
        self.step_outputs[subset_id].append(step_outputs)
        return loss

    def compute_output_losses(self,
                              outputs: ClassifierOutput,
                              labels: torch.Tensor,
                              subset_id: SubsetID) -> LossAggregator:
        self.compute_classification_loss(outputs.get_predictions(labels), labels, subset_id)

        if self.train_confidence:
            self.compute_confidence_loss(outputs.confidence, outputs.logits)

        if self.train_prototype_model:
            self.compute_prototype_loss(outputs.prototype_outputs, labels)

        return self.loss_aggregator

    def compute_classification_loss(self,
                                    predictions: torch.Tensor,
                                    labels: torch.Tensor,
                                    subset_id: SubsetID
                                    ) -> LossAggregator:
        use_label_smoothing = (self.label_smoothing > 0.0) and (subset_id == SubsetID.TRAIN)
        if use_label_smoothing:
            labels = self.apply_labels_smoothing(labels)

        loss_module = self.get_classification_loss_module()
        if isinstance(loss_module, nn.NLLLoss):
            predictions = torch.log(predictions)

        if self.multiclass:
            labels = torch.round(labels).to(torch.int64)

        base_loss = loss_module(predictions, labels)
        self.loss_aggregator.add_loss(base_loss, name="base_loss", loss_weight=self.base_loss_lambda)

        return self.loss_aggregator

    def get_classification_loss_module(self):
        if self.use_focal_loss:
            if self.train_confidence:
                raise NotImplementedError
            loss_module = FocalLoss(positive_class=self.positive_class)
        elif self.single_class:
            if self.train_confidence:
                loss_module = nn.BCELoss()
            else:
                loss_module = nn.BCEWithLogitsLoss()
        else:
            if self.train_confidence:
                loss_module = nn.NLLLoss()
            else:
                loss_module = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        return loss_module

    def apply_labels_smoothing(self,
                               labels: torch.Tensor,
                               label_smoothing: float = None
                               ) -> torch.Tensor:
        label_smoothing = self.label_smoothing if label_smoothing is None else label_smoothing
        if self.single_class:
            noise = torch.rand(*labels.shape, dtype=labels.dtype, device=labels.device) * label_smoothing
            noise = labels * - noise + (1.0 - labels) * noise
            labels = labels + noise
        return labels

    def compute_confidence_loss(self, confidence: torch.Tensor, logits: torch.Tensor) -> LossAggregator:
        confidence_loss = torch.mean(torch.log(torch.exp(-confidence) + 1.0))

        if self.confidence_budget is not None:
            self.update_confidence_lambda(confidence_loss.detach())

        self.loss_aggregator.add_loss(confidence_loss, "confidence_loss", self.confidence_lambda)

        if self.single_class:
            pearson = PearsonCorrCoef().to(logits.device)
            conf_corr = pearson(logits, confidence)
            # conf_corr_loss = torch.abs(conf_corr)
            conf_corr_loss = torch.square(conf_corr)
        else:
            raise NotImplementedError("Confidence correlation loss not yet implemented for multiclass case.")

        self.loss_aggregator.add_loss(conf_corr_loss, "confidence_correlation",
                                      self.confidence_lambda * self.confidence_corr_lambda)
        return self.loss_aggregator

    def compute_prototype_loss(self, prototype_outputs: PrototypeOutput, labels: torch.Tensor) -> None:
        distances = prototype_outputs.prototype_distances
        if self.prototype_model.class_specific:
            prototype_loss = self.prototype_model.compute_class_specific_loss(distances, labels)
        else:
            prototype_loss = self.prototype_model.compute_class_agnostic_loss(distances)
        self.loss_aggregator.add_loss(prototype_loss, "prototype_loss", self.prototype_lambda)

    # region Pack / Unpack
    @staticmethod
    def unpack_batch(batch) -> Union[tuple[list[torch.Tensor], torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        if len(batch) > 2:
            *inputs, labels = batch
        else:
            inputs, labels = batch
        return inputs, labels

    # endregion

    def update_confidence_lambda(self, confidence_loss: torch.Tensor) -> None:
        base_factor = 1.01
        increase_factor = base_factor
        decrease_factor = 1.0 / base_factor

        below_budget = (self.confidence_budget > confidence_loss).to(torch.float32)
        # if below budget: decrease lambda, else increase lambda
        update_factor = below_budget * decrease_factor + (1.0 - below_budget) * increase_factor

        self.confidence_lambda.multiply_(update_factor)

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        return self.base_step(batch, subset_id=SubsetID.TRAIN)

    # endregion

    # region Validation / Test
    def validation_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        with torch.no_grad():
            return self.base_step(batch, subset_id=SubsetID.VALIDATION)

    def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with torch.no_grad():
            return self.base_step(batch, subset_id=SubsetID.TEST)

    # endregion

    # region Classification metrics
    def log_epoch_outputs(self, subset: SubsetID):
        if self.trainer.sanity_checking:
            return {}

        outputs = self.step_outputs[subset]
        predictions, labels, confidence, loss = self.aggregate_epoch_outputs(outputs)
        labels = labels.to(torch.int)
        metrics = self.classification_summary(predictions, labels, confidence)

        # region Plot ROC / PR curves
        task = "binary" if self.single_class else "multiclass"
        num_classes = None if self.single_class else predictions.size(-1)

        # noinspection PyTypeChecker
        roc_curve = ROC(task=task, num_classes=num_classes)
        false_positive_rate, sensitivity, thresholds = roc_curve(predictions, labels)

        # noinspection PyTypeChecker
        pr_curve = PrecisionRecallCurve(task=task, num_classes=num_classes)
        precision, recall, thresholds = pr_curve(predictions, labels)

        if self.logger is not None:
            roc_plot_image = plot_line2d_to_array(false_positive_rate, sensitivity, output_size=(512, 512))
            pr_plot_image = plot_line2d_to_array(recall, precision, output_size=(512, 512))

            prefix = subset.as_prefix()
            self.logger.experiment.add_image(tag="{}_roc_curve".format(prefix),
                                             img_tensor=roc_plot_image,
                                             global_step=self.global_step, dataformats="HWC")
            self.logger.experiment.add_image(tag="{}_pr_curve".format(prefix),
                                             img_tensor=pr_plot_image,
                                             global_step=self.global_step, dataformats="HWC")
        # endregion

        if (loss is not None) and ("loss" not in metrics):
            metrics["loss"] = loss
        self.log_epoch_metrics(metrics, subset)

    def compute_confidence_threshold(self, trainer: pl.Trainer, data_loaders: list[DataLoader], update_threshold=True):
        confidence_threshold = self._compute_threshold(trainer, data_loaders, ConfidenceThreshold)
        if update_threshold:
            self.classification_summary.confidence_threshold = confidence_threshold
        return confidence_threshold

    def compute_eer_threshold(self, trainer: pl.Trainer, data_loaders: list[DataLoader], update_threshold=True):
        eer_threshold = self._compute_threshold(trainer, data_loaders, EERThreshold)
        if update_threshold:
            self.classification_summary.eer_threshold = eer_threshold
        return eer_threshold

    def compute_sensitivity_threshold(self, trainer: pl.Trainer, data_loaders: list[DataLoader], update_threshold=True):
        sensitivity_threshold = self._compute_threshold(trainer, data_loaders, self.sensitivity_threshold_class)
        if update_threshold:
            self.classification_summary.sensitivity_threshold = sensitivity_threshold
        return sensitivity_threshold

    def compute_available_thresholds(self,
                                     trainer: pl.Trainer,
                                     data_loaders: list[DataLoader],
                                     update_thresholds=True
                                     ) -> dict[type[ClassificationMetric], torch.Tensor]:
        available_thresholds = self.get_computable_thresholds()

        if len(available_thresholds) == 0:
            return {}

        thresholds = self._compute_thresholds(trainer, data_loaders, available_thresholds)

        if update_thresholds:
            if self.yield_confidence:
                self.classification_summary.confidence_threshold = thresholds[ConfidenceThreshold]

            if self.single_class or (self.classification_summary.positive_class is not None):
                self.classification_summary.eer_threshold = thresholds[EERThreshold]
                self.classification_summary.sensitivity_threshold = thresholds[self.sensitivity_threshold_class]

        return thresholds

    def get_computable_thresholds(self) -> list[type[ClassificationMetric]]:
        available_thresholds = []

        if self.yield_confidence:
            available_thresholds.append(ConfidenceThreshold)

        if self.single_class or (self.classification_summary.positive_class is not None):
            available_thresholds.append(EERThreshold)
            available_thresholds.append(self.sensitivity_threshold_class)

        return available_thresholds

    def _compute_threshold(self,
                           trainer: pl.Trainer,
                           data_loaders: list[DataLoader],
                           threshold_metric: type[ClassificationMetric]) -> torch.Tensor:
        return self._compute_thresholds(trainer, data_loaders, [threshold_metric])[threshold_metric]

    def _compute_thresholds(self,
                            trainer: pl.Trainer,
                            data_loaders: list[DataLoader],
                            threshold_metrics: list[type[ClassificationMetric]]
                            ) -> dict[type[ClassificationMetric], torch.Tensor]:
        classification_summary = ClassificationSummary(metrics=threshold_metrics,
                                                       num_thresholds=1000,
                                                       confidence_threshold=None,
                                                       eer_threshold=None)
        previous_summary = self.classification_summary
        self.classification_summary = classification_summary
        validation_results = trainer.validate(self, data_loaders, verbose=False)
        self.classification_summary = previous_summary

        thresholds = {}
        for threshold_metric in threshold_metrics:
            results_key = "validation_{}_epoch".format(threshold_metric.get_id())
            tmp = [loader_results[results_key] for loader_results in validation_results]
            threshold = torch.as_tensor(tmp).mean()
            thresholds[threshold_metric] = threshold

        return thresholds

    # endregion

    @staticmethod
    def aggregate_epoch_outputs(epoch_outputs: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
                                ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(epoch_outputs, dict):
            predictions = epoch_outputs["predictions"]
            labels = epoch_outputs["labels"]
            confidence = epoch_outputs.get("confidence")
            loss = epoch_outputs.get("loss")

        elif isinstance(epoch_outputs, list):
            # TODO: Try switching to logging all data_loaders independently
            if isinstance(epoch_outputs[0], list):
                epoch_outputs = [batch for loader_outputs in epoch_outputs for batch in loader_outputs]

            predictions = torch.cat([batch["predictions"] for batch in epoch_outputs])
            labels = torch.cat([batch["labels"] for batch in epoch_outputs])

            if all(["confidence" in batch for batch in epoch_outputs]):
                confidence = torch.cat([batch["confidence"] for batch in epoch_outputs])
            else:
                confidence = None

            if "loss" in epoch_outputs[0]:
                loss = torch.stack([batch["loss"] for batch in epoch_outputs])
            else:
                loss = None
        else:
            raise ValueError("`outputs` must either be a list or a dictionary.")

        return predictions, labels, confidence, loss

    @staticmethod
    def single_to_multiclass(predictions: torch.Tensor, negative_prevalence: float) -> torch.Tensor:
        classification_threshold = torch.quantile(predictions, q=negative_prevalence)
        predictions = (predictions - classification_threshold)
        above_threshold = (predictions > 0.0).float()
        positive_predictions = predictions * above_threshold
        negative_predictions = predictions * (above_threshold - 1.0)
        predictions = torch.stack([negative_predictions, positive_predictions], dim=-1)
        return predictions

    def get_saliency_maps(self,
                          inputs: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]]
                          ) -> Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]]:
        was_training = self.training
        self.eval()

        # scan : batch_size, channels, *spatial_dims
        inputs = set_require_grads(inputs)
        predictions = self(inputs)

        # incase we output confidence as well
        if isinstance(predictions, tuple):
            predictions, _ = predictions

        if len(predictions.shape) > 1:
            top_class = predictions.argmax(dim=-1)
            predictions = predictions[..., top_class]

        predictions.backward(torch.ones_like(predictions))
        gradients = get_gradients(inputs)

        self.train(was_training)
        return gradients

    # region Properties
    @property
    def logger(self) -> TensorBoardLogger:
        # noinspection PyTypeChecker
        return super(AbstractClassifier, self).logger

    @property
    @abstractmethod
    def yield_confidence(self) -> bool:
        raise NotImplementedError("`yield_confidence` must be implemented in subclasses")

    @property
    def train_confidence(self) -> bool:
        return self.yield_confidence

    @property
    def confidence_threshold(self) -> torch.Tensor | float | None:
        return self.classification_summary.confidence_threshold

    @property
    def has_confidence_threshold(self) -> bool:
        return self.classification_summary.confidence_threshold is not None

    @property
    def has_classification_thresholds(self) -> bool:
        return ((self.classification_summary.eer_threshold is not None) and
                (self.classification_summary.sensitivity_threshold is not None))

    @property
    def single_class(self) -> bool:
        return self.class_count == 1

    @property
    def multiclass(self) -> bool:
        return self.class_count > 1

    @property
    def outputs_features(self) -> bool:
        return (self.output_features_names is not None) and (len(self.output_features_names) > 0)

    @property
    def output_features_names(self) -> list[str] | None:
        return None

    @property
    def train_prototype_model(self) -> bool:
        return self.prototype_model is not None

    # endregion


# adapted from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 positive_class: int = -1,
                 ) -> None:
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.positive_class = positive_class

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2:
            batch_size, class_count = logits.shape
            positive_class = self.positive_class if (self.positive_class > 0) else (class_count + self.positive_class)
            is_positive = (labels == positive_class).float()

            log_probabilities = log_softmax(logits)
            loss = nll_loss(log_probabilities, labels, reduction="none")

            weighted_probabilities = torch.exp(log_probabilities[torch.arange(batch_size), labels])

        else:
            is_positive = labels.float()

            loss = binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")

            probabilities = torch.sigmoid(logits)
            weighted_probabilities = probabilities * labels + (1 - probabilities) * (1 - labels)

        loss = loss * ((1 - weighted_probabilities) ** self.gamma)

        if self.alpha >= 0:
            loss = loss * lerp(1 - self.alpha, self.alpha, is_positive)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
