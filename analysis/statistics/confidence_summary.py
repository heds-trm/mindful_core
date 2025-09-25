import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable

from mindful_core.data.subset_id import SubsetID
from mindful_core.data.data_folds import DataFold
from mindful_core.analysis.statistics.classification_summary import ConfidenceThreshold, EERThreshold, logits_to_probabilities
from mindful_core.utils.tensor_utils import linear_sample, get_kde, to_numpy

Classifier = Callable[[list[torch.Tensor] | torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class ConfidenceSummary(object):
    def __init__(self,
                 model: Classifier,
                 log_dir: str
                 ) -> None:
        self.model = model
        self.log_dir = Path(log_dir)

    def predict(self,
                data_loaders: dict[SubsetID, DataLoader]
                ) -> tuple[dict[SubsetID, torch.Tensor], dict[SubsetID, torch.Tensor], dict[SubsetID, torch.Tensor]]:
        probabilities = {}
        confidence_scores = {}
        labels = {}
        with torch.no_grad():
            for subset_id, data_loader in data_loaders.items():
                subset_confidence = []
                subset_labels = []
                subset_logits = []
                for *inputs, batch_labels in data_loader:
                    if len(inputs) == 1:
                        inputs = inputs[0]
                    batch_logits, batch_confidence = self.model(inputs)
                    subset_logits.append(batch_logits)
                    subset_confidence.append(batch_confidence)
                    subset_labels.append(batch_labels)
                probabilities[subset_id] = logits_to_probabilities(torch.concat(subset_logits, dim=0))
                confidence_scores[subset_id] = torch.concat(subset_confidence, dim=0)
                labels[subset_id] = torch.concat(subset_labels, dim=0).to(torch.int32)

        return probabilities, confidence_scores, labels

    @staticmethod
    def compute_thresholds(probabilities: dict[SubsetID, torch.Tensor],
                           confidence_scores: dict[SubsetID, torch.Tensor],
                           labels: dict[SubsetID, torch.Tensor],
                           subset_id=SubsetID.VALIDATION
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = probabilities[subset_id]
        labels = labels[subset_id]
        confidence_scores = confidence_scores[subset_id]
        confidence_threshold, _ = ConfidenceThreshold.compute_best_confidence(probabilities=probabilities,
                                                                              labels=labels,
                                                                              confidence=confidence_scores)
        in_distribution = confidence_scores > confidence_threshold
        selected_probabilities = probabilities[in_distribution]
        selected_labels = labels[in_distribution]
        probability_threshold = EERThreshold.compute_threshold(selected_probabilities, selected_labels)
        return confidence_threshold, probability_threshold

    @staticmethod
    def summarize_subset(probabilities: torch.Tensor,
                         probability_threshold: torch.Tensor,
                         confidence_scores: torch.Tensor,
                         confidence_threshold: torch.Tensor
                         ) -> list[tuple[float, bool, float, bool]]:
        positives = probabilities >= probability_threshold
        in_distribution = confidence_scores >= confidence_threshold
        summary_data = [(float(probability), bool(positive), float(confidence_score), bool(selected))
                        for probability, positive, confidence_score, selected
                        in zip(probabilities, positives, confidence_scores, in_distribution)]
        return summary_data

    def plot_confidence(self,
                        probabilities: torch.Tensor,
                        probability_threshold: torch.Tensor,
                        confidence_scores: torch.Tensor,
                        confidence_threshold: torch.Tensor,
                        labels: torch.Tensor,
                        prefix: str = None) -> None:
        positives = probabilities >= probability_threshold
        correct_predictions = positives == labels
        freq_figure = ConfidenceSummary.plot_confidence_kde(confidence_scores, correct_predictions,
                                                            confidence_threshold, plot_frequency=True)
        dens_figure = ConfidenceSummary.plot_confidence_kde(confidence_scores, correct_predictions,
                                                            confidence_threshold, plot_frequency=False)
        prefix = "" if prefix is None else prefix + "_"
        freq_figure.savefig((self.log_dir / "{}confidence_by_correctness_frequency.png".format(prefix)).as_posix())
        dens_figure.savefig((self.log_dir / "{}confidence_by_correctness_density.png".format(prefix)).as_posix())

    @staticmethod
    def plot_confidence_kde(confidence_scores: torch.Tensor,
                            correct_predictions: torch.Tensor,
                            confidence_threshold: torch.Tensor,
                            plot_frequency=False,
                            figure=None) -> plt.Figure:
        axis: plt.Axes
        if figure is None:
            figure, axis = plt.subplots()
        else:
            axis = figure.axes[0]

        confidence_scores = to_numpy(confidence_scores)
        confidence_correct: np.ndarray = confidence_scores[correct_predictions]
        confidence_incorrect: np.ndarray = confidence_scores[~correct_predictions]

        kde_correct = get_kde(confidence_correct)
        kde_incorrect = get_kde(confidence_incorrect)

        axis.set_xlabel("Confidence")
        if plot_frequency:
            axis.set_ylabel("Frequency")
            kde_correct *= len(confidence_correct) / np.nanmax(kde_correct)
            kde_incorrect *= len(confidence_incorrect) / np.nanmax(kde_incorrect)
        else:
            axis.set_ylabel("Density")

        indices = linear_sample(confidence_scores)

        axis.plot(indices, kde_correct, color="green", label="Correct predictions")
        axis.fill_between(indices, kde_correct, step="pre", alpha=0.1, color="green")
        axis.plot(indices, kde_incorrect, color="red", label="Incorrect predictions")
        axis.fill_between(indices, kde_incorrect, step="pre", alpha=0.1, color="red")

        ymax = max(np.nanmax(kde_correct), np.nanmax(kde_incorrect))
        axis.vlines(float(confidence_threshold), ymin=0.0, ymax=ymax, colors="blue")
        axis.legend()

        return figure

    def write_summary(self,
                      probabilities: dict[SubsetID, torch.Tensor],
                      probability_threshold: torch.Tensor,
                      confidence_scores: dict[SubsetID, torch.Tensor],
                      confidence_threshold: torch.Tensor,
                      labels: dict[SubsetID, torch.Tensor],
                      fold: DataFold,
                      prefix: str = None,
                      ):
        summary_data = sum([self.summarize_subset(probabilities[subset_id], probability_threshold,
                                                  confidence_scores[subset_id], confidence_threshold)
                            for subset_id in fold.samples], [])
        header = ["Probability", "Positive", "Confidence", "InDistribution"]
        summary = pd.DataFrame(summary_data, columns=header)

        fold_data_frame = fold.to_data_frame()
        summary = fold_data_frame.join(summary)

        base_filename = "confidence_analysis.csv" if prefix is None else "{}_confidence_analysis.csv".format(prefix)
        filepath = self.log_dir / base_filename
        summary.to_csv(filepath.as_posix(), index=False)

        self.plot_confidence(probabilities[SubsetID.TEST],
                             probability_threshold,
                             confidence_scores[SubsetID.TEST],
                             confidence_threshold=confidence_threshold,
                             labels=labels[SubsetID.TEST],
                             prefix=prefix)

    def __call__(self, data_loaders: dict[SubsetID, DataLoader], fold: DataFold) -> None:
        probabilities, confidence_scores, labels = self.predict(data_loaders)
        confidence_threshold, val_probability_threshold = self.compute_thresholds(probabilities,
                                                                                  confidence_scores,
                                                                                  labels)

        _, test_probability_threshold = self.compute_thresholds(probabilities,
                                                                confidence_scores,
                                                                labels,
                                                                subset_id=SubsetID.TEST)

        self.write_summary(probabilities=probabilities,
                           probability_threshold=val_probability_threshold,
                           confidence_scores=confidence_scores,
                           confidence_threshold=confidence_threshold,
                           labels=labels,
                           fold=fold)

        self.write_summary(probabilities=probabilities,
                           probability_threshold=test_probability_threshold,
                           confidence_scores=confidence_scores,
                           confidence_threshold=confidence_threshold,
                           labels=labels,
                           fold=fold,
                           prefix="test_positive_threshold")
