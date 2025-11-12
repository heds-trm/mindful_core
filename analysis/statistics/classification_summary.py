import torch
from torchmetrics.functional.classification.auroc import binary_auroc, multiclass_auroc
from torchmetrics import PearsonCorrCoef
import re
from abc import ABC, abstractmethod
from typing import Any, Union, Container, Optional

from mindful_core.utils.data_constants import LABEL

_classification_metric_id_regex = re.compile(r"((?<=[a-z\d])[A-Z]|(?!^)[A-Z](?=[a-z]))")


# region Metrics
class ClassificationMetric(ABC):
    @classmethod
    @abstractmethod
    def get_required_metrics(cls) -> list[Union[type["ClassificationMetric"], str]]:
        raise NotImplementedError

    @classmethod
    def get_required_metrics_ids(cls) -> list[str]:
        return [required_metric.get_id()
                if (isinstance(required_metric, type) and issubclass(required_metric, ClassificationMetric))
                else required_metric
                for required_metric in cls.get_required_metrics()]

    @classmethod
    @abstractmethod
    def requires_thresholds(cls) -> bool:
        raise NotImplementedError("Must be implemented in subclasses.")

    @classmethod
    def requires_confidence(cls) -> bool:
        return "confidence" in cls.get_required_metrics()

    @classmethod
    def get_missing_metrics(cls, computed_metrics: Container[str]
                            ) -> list[type["ClassificationMetric"]]:
        missing_metrics = []
        for required_metric in cls.get_required_metrics():
            if isinstance(required_metric, type) and issubclass(required_metric, ClassificationMetric):
                required_metric_id = required_metric.get_id()
                if (required_metric_id not in computed_metrics) and (required_metric not in missing_metrics):
                    missing_metrics.append(required_metric)
        return missing_metrics

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.compute_metric(*args, **kwargs)

    @abstractmethod
    def compute_metric(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def get_id(cls) -> str:
        return _classification_metric_id_regex.sub(r"_\1", cls.__name__).lower()


# region ROC (AUC)
def compute_auroc(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.int32)

    if labels.dtype == torch.bool:
        labels = labels.to(torch.int)

    class_count = probabilities.size(-1) if len(probabilities.shape) > 1 else 1
    if (len(probabilities.shape) == 1) or (class_count == 1):
        auroc = binary_auroc(probabilities, labels)
    else:
        int_labels = torch.round(labels).to(torch.int64)
        auroc = multiclass_auroc(probabilities, int_labels, num_classes=class_count)

    return auroc


class AUROC(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[str]:
        return ["probabilities", "labels"]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return compute_auroc(probabilities, labels)


# region Confidence Threshold

class AboveThreshold(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[Union[type["ClassificationMetric"], str]]:
        return ["probabilities", "labels", "confidence", AUROC, "confidence_threshold"]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, probabilities: torch.Tensor, labels: torch.Tensor, confidence: torch.Tensor,
                       auroc: torch.Tensor, confidence_threshold: torch.Tensor) -> torch.Tensor:
        if confidence_threshold is None:
            confidence_threshold, * \
                _ = ConfidenceThreshold.compute_best_confidence(probabilities, labels, confidence, auroc)

        above_threshold = torch.greater(confidence, confidence_threshold)
        above_threshold = above_threshold.float().mean()
        return above_threshold


class ConfidenceThreshold(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[Union[type["ClassificationMetric"], str]]:
        return ["probabilities", "labels", "confidence", AUROC]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, probabilities: torch.Tensor, labels: torch.Tensor, confidence: torch.Tensor,
                       auroc: torch.Tensor) -> torch.Tensor:
        best_threshold, *_ = self.compute_best_confidence(probabilities, labels, confidence, auroc)
        return best_threshold

    @staticmethod
    def compute_best_confidence(probabilities: torch.Tensor, labels: torch.Tensor, confidence: torch.Tensor,
                                reference: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """ Computes the confidence threshold that maximizes the AUROC for the given probabilities and labels
        ### Parameters
        1. probabilities
            - A 1D tensor. Shape [sample_count]. Contains probabilities comprised between 0 and 1.
        2. labels
            - A 1D tensor. Shape [sample_count]. Contains labels for given probabilities.
        3. confidence
            - A 1D tensor. Shape [sample_count]. Contains confidence scores for given probabilities.
        4. reference (Optional)
            - A scalar. The reference AUROC value for all samples. Computed if not provided.

        ### Returns
        - best_confidence: A scalar. The confidence threshold maximizing the AUROC.
        - best_auroc: A scalar. The AUROC at the returned confidence threshold.
        """
        sample_count = confidence.size(0)
        max_threshold_count = sample_count // 2
        confidence, confidence_indices = torch.sort(confidence)
        probabilities = probabilities[confidence_indices]
        labels = labels[confidence_indices]
        if reference is None:
            reference = compute_auroc(probabilities, labels)

        best_auc = reference
        best_index = 0
        for i in range(1, max_threshold_count):
            partial_labels = labels[i:]
            if torch.all(partial_labels) or not torch.any(partial_labels):
                break

            partial_auc = compute_auroc(probabilities[i:], partial_labels)
            if partial_auc > best_auc:
                best_auc = partial_auc
                best_index = i

        return confidence[best_index], best_auc


# endregion
# endregion


class CorrectPredictions(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[str]:
        return ["predictions", "labels"]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if len(predictions.shape) == 1:
            labels = labels.to(torch.int64)
        else:
            labels = labels.to(torch.bool).unsqueeze(dim=0)
        return torch.eq(predictions, labels)


# region Accuracy (base)
class Accuracy(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [CorrectPredictions]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, correct_predictions: torch.Tensor) -> torch.Tensor:
        return correct_predictions.to(torch.float).mean(-1)


class AccuracyMaxIndex(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Accuracy]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, accuracy: torch.Tensor) -> torch.Tensor:
        return torch.argmax(accuracy)


class AccuracyMax(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Accuracy]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, accuracy: torch.Tensor) -> torch.Tensor:
        return torch.max(accuracy)


class AccuracyAverage(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Accuracy]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, accuracy: torch.Tensor) -> torch.Tensor:
        return accuracy.mean()


# endregion

# region Sensitivity = Recall / Specificity / Precision (base)
class Sensitivity(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [CorrectPredictions, "predictions", "labels", "class_count"]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self,
                       correct_predictions: torch.Tensor,
                       predictions: torch.Tensor,
                       labels: torch.Tensor,
                       class_count: int,
                       ) -> torch.Tensor:

        if class_count == 1:
            labels = labels.to(torch.bool).unsqueeze(dim=0)
            true_positives = torch.logical_and(labels, correct_predictions)
            labeled_positives_count = labels.to(torch.float).sum()
            true_positives_count = true_positives.to(torch.float).sum(dim=-1)

        else:
            labels = labels.to(torch.int64)
            true_positives = predictions[correct_predictions]
            labeled_positives_count = labels.bincount(minlength=class_count)
            true_positives_count = true_positives.bincount(minlength=class_count)

        sensitivity = true_positives_count / labeled_positives_count

        if class_count == 2:
            sensitivity = sensitivity[1]

        return sensitivity


Recall = Sensitivity


class Specificity(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [CorrectPredictions, "predictions", "labels", "class_count"]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self,
                       correct_predictions: torch.Tensor,
                       predictions: torch.Tensor,
                       labels: torch.Tensor,
                       class_count: int,
                       ) -> torch.Tensor:
        if class_count == 1:
            labels = labels.to(torch.bool).unsqueeze(dim=0)
            real_negatives = ~labels
            real_negatives_count = real_negatives.to(torch.float).sum()

            true_negatives = torch.logical_and(real_negatives, correct_predictions)
            true_negatives_count = true_negatives.to(torch.float).sum(dim=-1)

        else:
            classes = torch.as_tensor(list(range(class_count)), device=predictions.device)
            classes = classes.unsqueeze(0)

            not_labels = torch.not_equal(labels.unsqueeze(1), classes)
            not_predictions = torch.not_equal(predictions.unsqueeze(1), classes)
            true_negatives = torch.logical_and(not_labels, not_predictions)

            real_negatives_count = not_labels.to(torch.int64).sum(0)
            true_negatives_count = true_negatives.to(torch.int64).sum(0)

        specificity = true_negatives_count / real_negatives_count

        if class_count == 2:
            specificity = specificity[1]

        return specificity


class Precision(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [CorrectPredictions, "labels"]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, correct_predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if len(correct_predictions.shape) == 2:
            labels = labels.to(torch.bool).unsqueeze(dim=0)

        elif len(correct_predictions.shape) != 1:
            raise ValueError(correct_predictions.shape)

        true_positives = torch.logical_and(labels, correct_predictions)
        true_positives_count = true_positives.to(torch.float).sum(dim=-1)
        sample_count = labels.size(-1)
        return true_positives_count / sample_count


# region Averages (sensitivity, specificity, precision)

class SensitivityAverage(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Sensitivity]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, sensitivity: torch.Tensor) -> torch.Tensor:
        return sensitivity.mean()


class SpecificityAverage(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Specificity]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, specificity: torch.Tensor) -> torch.Tensor:
        return specificity.mean()


# endregion
# endregion

# region False Positive/Negative Rates
class FPR(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Specificity]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, specificity: torch.Tensor) -> torch.Tensor:
        return 1.0 - specificity


class FNR(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Sensitivity]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, sensitivity: torch.Tensor) -> torch.Tensor:
        return 1.0 - sensitivity


# endregion

# region EER (base)
class EERIndex(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [FPR, FNR]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, fpr: torch.Tensor, fnr: torch.Tensor) -> torch.Tensor:
        return torch.argmin(torch.abs(fnr - fpr))


class EER(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [FPR, FNR, EERIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, fpr: torch.Tensor, fnr: torch.Tensor, eer_index: torch.Tensor) -> torch.Tensor:
        eer_fpr = fpr[eer_index]
        eer_fnr = fnr[eer_index]
        return (eer_fpr + eer_fnr) * 0.5


class EERThreshold(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[Union[type["ClassificationMetric"], str]]:
        return ["probabilities", "labels"]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    @staticmethod
    def compute_threshold(probabilities: torch.Tensor,
                          labels: torch.Tensor,
                          num_thresholds: int = 1000
                          ) -> torch.Tensor:
        class_count = get_class_count(probabilities)
        predictions, thresholds = probabilities_to_predictions(probabilities, num_thresholds)
        input_data = {"predictions": predictions, "labels": labels, "class_count": class_count}
        eer_index = ClassificationSummary.compute_metrics(input_data, metrics=[EERIndex])["eer_index"]
        return thresholds[eer_index]

    def compute_metric(self,
                       probabilities: torch.Tensor,
                       labels: torch.Tensor,
                       num_thresholds: int = 1000
                       ) -> torch.Tensor:
        return self.compute_threshold(probabilities, labels, num_thresholds)


# endregion

# region F-score

class F1Score(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type[ClassificationMetric]]:
        return [Precision, Sensitivity]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, precision: torch.Tensor, sensitivity: torch.Tensor) -> torch.Tensor:
        return 2.0 * (precision * sensitivity) / (precision + sensitivity + 1e-7)


class F1ScoreAverage(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type[ClassificationMetric]]:
        return [F1Score]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, f1_score: torch.Tensor) -> torch.Tensor:
        return f1_score.mean()


# endregion

# region At EER
class AccuracyAtEER(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Accuracy, EERIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, accuracy: torch.Tensor, eer_index: torch.Tensor) -> torch.Tensor:
        return accuracy[eer_index]


class SensitivityAtEER(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Sensitivity, EERIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, sensitivity: torch.Tensor, eer_index: torch.Tensor) -> torch.Tensor:
        return sensitivity[eer_index]


class SpecificityAtEER(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Specificity, EERIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, specificity: torch.Tensor, eer_index: torch.Tensor) -> torch.Tensor:
        return specificity[eer_index]


# endregion

# region At X% Sensitivity
def make_metrics_for_sensitivity_threshold(base_metrics: list[Union[str, type[ClassificationMetric]]],
                                           relative_threshold: float
                                           ) -> list[type[ClassificationMetric]]:
    threshold_id = int(relative_threshold * 100)
    sensitivity_index = type("SensitivityIndex{}".format(threshold_id),
                             (SensitivityIndex,),
                             {"threshold": relative_threshold})
    metrics = []
    if ("sensitivity_index" in base_metrics) or (SensitivityIndex in base_metrics):
        metrics.append(sensitivity_index)

    if ("sensitivity_threshold" in base_metrics) or (SensitivityThreshold in base_metrics):
        sensitivity_threshold = type("SensitivityThreshold{}".format(threshold_id),
                                     (SensitivityThreshold,),
                                     {"index_class": sensitivity_index})
        metrics.append(sensitivity_threshold)

    if ("accuracy_at_sensitivity" in base_metrics) or (AccuracyAtSensitivity in base_metrics):
        accuracy_at_sensitivity = type("AccuracyAtSensitivity{}".format(threshold_id),
                                       (AccuracyAtSensitivity,),
                                       {"index_class": sensitivity_index})
        metrics.append(accuracy_at_sensitivity)

    if ("sensitivity_at_sensitivity" in base_metrics) or (SensitivityAtSensitivity in base_metrics):
        sensitivity_at_sensitivity = type("SensitivityAtSensitivity{}".format(threshold_id),
                                          (SensitivityAtSensitivity,),
                                          {"index_class": sensitivity_index})
        metrics.append(sensitivity_at_sensitivity)

    if ("specificity_at_sensitivity" in base_metrics) or (SpecificityAtSensitivity in base_metrics):
        specificity_at_sensitivity = type("SpecificityAtSensitivity{}".format(threshold_id),
                                          (SpecificityAtSensitivity,),
                                          {"index_class": sensitivity_index})
        metrics.append(specificity_at_sensitivity)

    # noinspection PyTypeChecker
    return metrics


class SensitivityIndex(ClassificationMetric):
    threshold: float = None

    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Sensitivity]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, sensitivity: torch.Tensor) -> torch.Tensor:
        return torch.argmin(torch.abs(sensitivity - self.threshold))


class SensitivityThreshold(ClassificationMetric):
    index_class: type[SensitivityIndex] = None

    @classmethod
    def get_required_metrics(cls) -> list[Union[type["ClassificationMetric"], str]]:
        return ["probabilities", "labels"]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    @classmethod
    def compute_threshold(cls,
                          probabilities: torch.Tensor,
                          labels: torch.Tensor,
                          num_thresholds: int = 1000
                          ) -> torch.Tensor:
        if cls.index_class is None:
            raise RuntimeError()

        class_count = get_class_count(probabilities)
        predictions, thresholds = probabilities_to_predictions(probabilities, num_thresholds)
        input_data = {"predictions": predictions, "labels": labels, "class_count": class_count}
        metrics = ClassificationSummary.compute_metrics(input_data, metrics=[cls.index_class])
        sensitivity_index = metrics[cls.index_class.get_id()]
        return thresholds[sensitivity_index]

    def compute_metric(self,
                       probabilities: torch.Tensor,
                       labels: torch.Tensor,
                       num_thresholds: int = 1000
                       ) -> torch.Tensor:
        return self.compute_threshold(probabilities, labels, num_thresholds)


class AccuracyAtSensitivity(ClassificationMetric):
    index_class: SensitivityIndex = None

    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        if cls.index_class is None:
            raise RuntimeError()
        return [Accuracy, cls.index_class]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, accuracy: torch.Tensor, **kwargs) -> torch.Tensor:
        sensitivity_index = kwargs[self.index_class.get_id()]
        return accuracy[sensitivity_index]


class SensitivityAtSensitivity(ClassificationMetric):
    index_class: SensitivityIndex = None

    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        if cls.index_class is None:
            raise RuntimeError()
        return [Sensitivity, cls.index_class]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, sensitivity: torch.Tensor, **kwargs) -> torch.Tensor:
        sensitivity_index = kwargs[self.index_class.get_id()]
        return sensitivity[sensitivity_index]


class SpecificityAtSensitivity(ClassificationMetric):
    index_class: SensitivityIndex = None

    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        if cls.index_class is None:
            raise RuntimeError()
        return [Specificity, cls.index_class]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, specificity: torch.Tensor, **kwargs) -> torch.Tensor:
        sensitivity_index = kwargs[self.index_class.get_id()]
        return specificity[sensitivity_index]


# endregion

# region At max accuracy
class SensitivityAtAccuracyMax(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Sensitivity, AccuracyMaxIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, sensitivity: torch.Tensor, accuracy_max_index: torch.Tensor) -> torch.Tensor:
        return sensitivity[accuracy_max_index]


class SpecificityAtAccuracyMax(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Specificity, AccuracyMaxIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, specificity: torch.Tensor, accuracy_max_index: torch.Tensor) -> torch.Tensor:
        return specificity[accuracy_max_index]


# endregion

# region Youden Statistic

class YoudenStatisticIndex(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Sensitivity, FPR]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, sensitivity: torch.Tensor, fpr: torch.Tensor) -> torch.Tensor:
        return torch.argmax(sensitivity - fpr)


class YoudenThreshold(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return ["thresholds", YoudenStatisticIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, thresholds: torch.Tensor, youden_statistic_index: torch.Tensor) -> torch.Tensor:
        return thresholds[youden_statistic_index]


# endregion

# region At Youden Statistic
class AccuracyAtYouden(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Accuracy, YoudenStatisticIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, accuracy: torch.Tensor, youden_statistic_index: torch.Tensor) -> torch.Tensor:
        return accuracy[youden_statistic_index]


class SensitivityAtYouden(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Sensitivity, YoudenStatisticIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, sensitivity: torch.Tensor, youden_statistic_index: torch.Tensor) -> torch.Tensor:
        return sensitivity[youden_statistic_index]


class SpecificityAtYouden(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [Specificity, YoudenStatisticIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, specificity: torch.Tensor, youden_statistic_index: torch.Tensor) -> torch.Tensor:
        return specificity[youden_statistic_index]


class F1ScoreAtYouden(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[type["ClassificationMetric"]]:
        return [F1Score, YoudenStatisticIndex]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return True

    def compute_metric(self, f1_score: torch.Tensor, youden_statistic_index: torch.Tensor) -> torch.Tensor:
        return f1_score[youden_statistic_index]


# endregion

class ConfidencePositivityCorrelation(ClassificationMetric):
    @classmethod
    def get_required_metrics(cls) -> list[str]:
        return ["probabilities", "confidence"]

    @classmethod
    def requires_thresholds(cls) -> bool:
        return False

    def compute_metric(self, probabilities: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        pearson = PearsonCorrCoef().to(probabilities.device)
        return pearson(probabilities, confidence)


# endregion

class ClassificationSummary(object):
    def __init__(self,
                 metrics: list[type["ClassificationMetric"]],
                 num_thresholds=1000,
                 confidence_threshold: Union[torch.Tensor, float] = None,
                 eer_threshold: Union[torch.Tensor, float] = None,
                 sensitivity_threshold: Union[torch.Tensor, float] = None,
                 positive_class=-1,
                 sum_multiclass_logits: bool = False,
                 ) -> None:
        self.metrics = metrics
        self.num_thresholds = num_thresholds
        self.confidence_threshold = confidence_threshold
        self.eer_threshold = eer_threshold
        self.sensitivity_threshold = sensitivity_threshold
        self.positive_class = positive_class
        self.sum_multiclass_logits = sum_multiclass_logits

    def __call__(self,
                 logits: torch.Tensor,
                 labels: torch.Tensor,
                 confidence: torch.Tensor = None,
                 probabilities: torch.Tensor = None,
                 ) -> dict[str, torch.Tensor]:

        data = self.prepare_data(logits, labels, confidence, probabilities)
        summary = self.compute_metrics(data, self.metrics)

        if (self.confidence_threshold is not None) and (confidence is not None):
            partial_summary = self.call_partial(logits, labels, confidence)
            summary = {**summary, **partial_summary}

        return summary

    def call_partial(self, logits: torch.Tensor,
                     labels: torch.Tensor,
                     confidence: torch.Tensor) -> dict[str, torch.Tensor]:
        partial_data = self.prepare_data(**self.filter_samples_by_confidence(logits, labels, confidence))

        metrics_partial = [metric for metric in self.metrics if not metric.requires_confidence()]
        summary = self.compute_metrics(partial_data, metrics=metrics_partial)
        summary = {"partial_{}".format(key): value for key, value in summary.items()}

        return summary

    def prepare_data(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor,
                     confidence: torch.Tensor = None,
                     probabilities: torch.Tensor = None,
                     ) -> dict[str, Union[torch.Tensor, int]]:
        class_count = get_class_count(logits)
        positive_multiclass = (class_count > 1) and (self.positive_class is not None)
        if self.sum_multiclass_logits and positive_multiclass:
            weights = torch.ones([class_count, class_count], dtype=logits.dtype, device=logits.device) * -1.0
            # weights[:, self.positive_class] = - 1.0 / (class_count - 1)
            # weights[self.positive_class, self.positive_class] = 1.0
            weights += torch.eye(class_count, dtype=logits.dtype, device=logits.device) * 2.0
            logits = logits @ weights
            
        if probabilities is None:
            probabilities = logits_to_probabilities(logits)

        if positive_multiclass:
            positive_class = self.positive_class if (self.positive_class >= 0) else (class_count + self.positive_class)
            probabilities = probabilities[..., positive_class]

            labels = (labels == positive_class).to(labels.dtype)
            class_count = 1

        data: dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": probabilities,
            "labels": labels,
            "class_count": class_count,
        }

        predictions, thresholds = probabilities_to_predictions(probabilities, self.num_thresholds)
        data["predictions"] = predictions
        if thresholds is not None:
            data["thresholds"] = thresholds

        if confidence is not None:
            data["confidence"] = confidence

        if ConfidenceThreshold not in self.metrics:
            data["confidence_threshold"] = self.confidence_threshold

        return data

    @staticmethod
    def compute_metrics(data: dict[str, torch.Tensor],
                        metrics: list[type[ClassificationMetric]],
                        ) -> dict[str, torch.Tensor]:
        metrics_outputs = {}
        metrics_to_compute = ClassificationSummary.get_metrics_to_compute(data, metrics)

        while len(metrics_to_compute) > 0:
            current_metric = metrics_to_compute[0]
            missing_metrics_for_current = current_metric.get_missing_metrics(data)
            if len(missing_metrics_for_current) > 0:
                metrics_to_compute = missing_metrics_for_current + metrics_to_compute
            else:
                metrics_to_compute.pop(0)
                metric_id = current_metric.get_id()
                metric_inputs = {required_metric: data[required_metric]
                                 for required_metric in current_metric.get_required_metrics_ids()}
                metric_output = current_metric()(**metric_inputs)
                if (current_metric in metrics) and (len(metric_output.shape) == 0):
                    metrics_outputs[metric_id] = metric_output
                data[metric_id] = metric_output

        return metrics_outputs

    @staticmethod
    def get_metrics_to_compute(data: dict[str, torch.Tensor],
                               metrics: list[type[ClassificationMetric]]
                               ) -> list[type[ClassificationMetric]]:
        return [metric for metric in metrics if
                (
                        not (metric.requires_thresholds() and is_multi_class(data)) and
                        ("confidence" in data or not metric.requires_confidence())
                )
                ]

    def filter_samples_by_confidence(self,
                                     logits: torch.Tensor,
                                     labels: torch.Tensor,
                                     confidence: torch.Tensor
                                     ) -> dict[str, torch.Tensor]:
        if self.confidence_threshold is None:
            raise RuntimeError("The confidence threshold is unknown, but is required to filter samples by confidence.")

        above_threshold = confidence >= self.confidence_threshold

        logits = logits[above_threshold]
        labels = labels[above_threshold]
        confidence = confidence[above_threshold]
        return {
            "logits": logits,
            "labels": labels,
            "confidence": confidence
        }

    def get_available_inferences(self,
                                 probabilities: torch.Tensor,
                                 labels: torch.Tensor,
                                 ) -> dict[str, torch.Tensor]:
        inferences: dict[str, torch.Tensor] = {}
        multi_class = len(probabilities.shape) > 1

        if multi_class and (self.positive_class is None):
            probabilities, predicted_class = torch.max(probabilities, dim=-1)
            predicted_class = predicted_class.to(torch.int32)
            inferences[LABEL] = labels
            inferences["Probability"] = probabilities.to(torch.float32)

            inferences["Predicted Class"] = predicted_class
            inferences["Correct Prediction"] = predicted_class == labels
        else:
            if multi_class:
                if self.positive_class < 0:
                    ref_label = probabilities.shape[-1] + self.positive_class
                else:
                    ref_label = self.positive_class
                probabilities = probabilities[..., self.positive_class]
                labels = (labels == ref_label).to(torch.int32)
            inferences[LABEL] = labels
            inferences["Probability"] = probabilities.to(torch.float32)

            # region EER
            if self.eer_threshold is not None:
                prediction_eer = probabilities > self.eer_threshold
                prediction_eer = prediction_eer.to(torch.int32)
                inferences["Prediction (EER)"] = prediction_eer
                inferences["Correctness (EER)"] = prediction_eer == labels
                inferences["Threshold (EER)"] = torch.ones_like(probabilities) * self.eer_threshold
            # endregion

            # region Sensitivity
            if self.sensitivity_threshold is not None:
                prediction_sens = probabilities > self.sensitivity_threshold
                prediction_sens = prediction_sens.to(torch.int32)
                inferences["Prediction (Sens.)"] = prediction_sens
                inferences["Correctness (Sens.)"] = prediction_sens == labels
                inferences["Threshold (Sens.)"] = torch.ones_like(probabilities) * self.sensitivity_threshold
            # endregion
        return inferences


def logits_to_probabilities(logits: torch.Tensor, softmax_dim=-1) -> torch.Tensor:
    if len(logits.shape) > 1:
        probabilities = torch.softmax(logits, dim=softmax_dim)
    else:
        probabilities = torch.sigmoid(logits)
    return probabilities


def probabilities_to_predictions(probabilities: torch.Tensor,
                                 num_thresholds: int
                                 ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    epsilon = 1e-5
    if not is_multi_class(probabilities):
        # region Binary predictions from logits and thresholds
        thresholds = torch.linspace(epsilon, 1.0 - epsilon,
                                    steps=num_thresholds,
                                    device=probabilities.device)

        probabilities_min, _ = probabilities.min(dim=0)
        probabilities_max, _ = probabilities.max(dim=0)
        predictions = (probabilities - probabilities_min) / (probabilities_max - probabilities_min)

        predictions = predictions.unsqueeze(dim=0)
        predictions: torch.Tensor = torch.greater(predictions, thresholds.unsqueeze(dim=1))
        # endregion
    else:
        thresholds = None
        predictions = probabilities.argmax(dim=-1)

    return predictions, thresholds


def get_class_count(logits_or_probabilities: torch.Tensor) -> int:
    if len(logits_or_probabilities.shape) > 1:
        class_count = logits_or_probabilities.size(-1)
    else:
        class_count = 1
    return class_count


def is_multi_class(data: Union[dict[str, torch.Tensor], torch.Tensor]) -> bool:
    if isinstance(data, dict):
        if "class_count" in data:
            return data["class_count"] > 1
        elif "logits" in data:
            return is_multi_class(data["logits"])
        elif "predictions" in data:
            return data["predictions"].dtype != torch.bool
        else:
            raise ValueError("`data` must either be a tensor, or contain logits, predictions or class_count.")
    else:
        return len(data.shape) > 1
