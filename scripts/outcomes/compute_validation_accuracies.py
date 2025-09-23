import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

from utils.misc import load_json, write_json
from models.classification.ensemble.ensemble_classifier import EnsembleClassifier


def get_validation_accuracy(model_path: Path, threshold_name: str) -> float:
    inferences_path = model_path / "inferences.csv"
    inferences = pd.read_csv(inferences_path, index_col="ScanID")
    is_validation = inferences["SubsetID"] == "validation"
    inferences = inferences[is_validation]

    correctness_column = "Correctness ({})".format(threshold_name)
    correctness = inferences[correctness_column]

    return correctness.mean()


def update_ensemble_hparams(ensemble_hparams: dict, threshold_name: str) -> dict:
    models_paths = EnsembleClassifier.find_mindful_models(ensemble_hparams["models_paths"])
    accuracies = {model_path: get_validation_accuracy(model_path, threshold_name) for model_path in models_paths}

    ensemble_hparams["models_paths"] = [model_path.as_posix() for model_path in accuracies.keys()]
    ensemble_hparams["accuracies"] = [accuracy for accuracy in accuracies.values()]

    return ensemble_hparams


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--root")
    arg_parser.add_argument("--threshold_name", default="EER")

    args = arg_parser.parse_args()
    ensembles_hparams_root = Path(args.root)
    threshold_name: str = args.threshold_name

    ensemble_hparams_paths = list(ensembles_hparams_root.rglob("*.json"))
    print("Updating the following model hparam files: {}".format(ensemble_hparams_paths))

    for ensemble_hparams_path in tqdm(ensemble_hparams_paths):
        ensemble_hparams = load_json(ensemble_hparams_path)
        if "models_paths" not in ensemble_hparams:
            print("{} is not ensemble model hparam file".format(ensemble_hparams_path))
            continue

        if "accuracies" in ensemble_hparams:
            print("{} already contains accuracies, skipping".format(ensemble_hparams_path))
            continue

        ensemble_hparams = update_ensemble_hparams(ensemble_hparams, threshold_name)
        write_json(ensemble_hparams_path, ensemble_hparams)


if __name__ == "__main__":
    main()
