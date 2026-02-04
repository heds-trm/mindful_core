import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from mindful_core.analysis.statistics.classification_summary import ConfidenceThreshold


def get_partial_auroc_inputs(confidence_analysis: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = confidence_analysis["Probability"]
    labels = confidence_analysis["Label"]
    confidence = confidence_analysis["Confidence"]

    probabilities = torch.as_tensor(probabilities.tolist())
    labels = torch.as_tensor(labels.tolist())
    confidence = torch.as_tensor(confidence.tolist())

    return probabilities, labels, confidence


def plot_partial_aurocs(confidence_analysis_path: Path) -> plt.Figure:
    confidence_analysis = pd.read_csv(confidence_analysis_path, index_col="ScanID")
    
    subset = confidence_analysis["SubsetID"]
    is_validation = subset == "validation"
    validation_data = confidence_analysis[is_validation]
    val_probabilities, val_labels, val_confidence = get_partial_auroc_inputs(validation_data)

    is_test = subset == "test"
    test_data = confidence_analysis[is_test]
    test_probabilities, test_labels, test_confidence = get_partial_auroc_inputs(test_data)

    confidence_threshold, _, = ConfidenceThreshold.compute_best_confidence(val_probabilities, val_labels, val_confidence)
    sorted_confidence, aurocs = ConfidenceThreshold.compute_partial_aurocs(test_probabilities, test_labels, test_confidence)

    confidence_threshold = confidence_threshold.cpu().numpy()
    sorted_confidence = sorted_confidence.cpu().numpy()
    test_confidence = test_confidence.cpu().numpy()
    aurocs = aurocs.cpu().numpy() * 100.0

    kept: np.ndarray = np.expand_dims(test_confidence, axis=0) >= np.expand_dims(sorted_confidence, axis=1)
    kept = kept.astype(np.float32).mean(axis=1) * 100.0

    figure, axis = plt.subplots()
    axis.plot(sorted_confidence, aurocs, color="green", label="AUROC (%)")
    axis.plot(sorted_confidence, kept, color="red", label="Kept (%)")
    axis.vlines(confidence_threshold, ymin=0.0, ymax=aurocs.max(), colors="blue")
    axis.set_xlabel("Confidence")
    axis.legend()

    return figure


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("root")

    args = arg_parser.parse_args()
    root = Path(args.root)

    sns.set_theme()

    confidence_analysis_paths = list(root.rglob("confidence_analysis.csv"))
    for confidence_analysis_path in tqdm(confidence_analysis_paths):
        figure = plot_partial_aurocs(confidence_analysis_path)
        save_path = confidence_analysis_path.parent / "partial_aurocs.png"
        figure.savefig(save_path.as_posix())
        plt.close(figure)


if __name__ == "__main__":
    main()
