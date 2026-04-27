import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from monai.utils import set_determinism
from pytorch_lightning import seed_everything
import numpy as np
from sklearn import svm, ensemble, linear_model
import pandas as pd
import argparse
from pathlib import Path
import os
from typing import Any

from mindful_core.data import SubsetID
from mindful_core.data.data_folds import PresetFold
from mindful_core.data.transforms.vector_data import ScalarPreprocess, CategoricalPreprocess
from mindful_core.analysis.statistics.classification_summary import (
    ClassificationSummary,

    AUROC,

    EERThreshold,
    EER,
    AccuracyAtEER,
    SpecificityAtEER,
    SensitivityAtEER,

    make_metrics_for_sensitivity_threshold,
    SensitivityThreshold,
    AccuracyAtSensitivity,
    SensitivityAtSensitivity,
    SpecificityAtSensitivity,

    AccuracyAverage,
    SensitivityAverage,
    SpecificityAverage,
    F1ScoreAverage
)
from mindful_core.scripts.outcomes.gather_test_results import (
    summarize_experiments,
    sort_experiments_summary_columns,
    format_experiments_summary
)


# region Neural Networks
class ResidualLinear(nn.Module):
    def __init__(self, width: int,
                 depth: int,
                 use_bias=True,
                 dropout=0.0,
                 pre_activate=True):
        super(ResidualLinear, self).__init__()
        self.width = width
        self.depth = depth
        self.use_bias = use_bias
        self.pre_activate = pre_activate

        self.linear_layers_0 = [self._make_linear() for _ in range(depth)]
        self.linear_layers_1 = [self._make_linear() for _ in range(depth)]
        if self.use_bias:
            self.biases_0 = [self._make_bias() for _ in range(depth)]
            self.biases_1 = [self._make_bias() for _ in range(depth)]
            self.biases_2 = [self._make_bias() for _ in range(depth)]
            self.biases_3 = [self._make_bias() for _ in range(depth)]
        else:
            self.biases_0, self.biases_1, self.biases_2, self.biases_3 = None, None, None, None
        self.multipliers = [nn.Parameter(torch.as_tensor((1.0,)), requires_grad=True)
                            for _ in range(depth)]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.pre_activate:
            inputs = self.relu(inputs)

        for i in range(self.depth):
            inputs = self._apply_block(inputs, i)
        return inputs

    def _make_linear(self):
        weights = torch.zeros((self.width, self.width))
        weights = nn.Parameter(weights, requires_grad=True)
        nn.init.kaiming_uniform_(weights, a=np.sqrt(5))
        return weights

    def _make_bias(self):
        weights = torch.zeros((1, self.width))
        return nn.Parameter(weights, requires_grad=True)

    def _apply_block(self, inputs: torch.Tensor, block_index: int) -> torch.Tensor:
        # FixUp: https://arxiv.org/pdf/1901.09321.pdf
        residual = inputs

        if self.use_bias:
            inputs = inputs + self.biases_0[block_index]
        inputs = inputs @ self.linear_layers_0[block_index]
        if self.use_bias:
            inputs = inputs + self.biases_1[block_index]
        inputs = self.relu(inputs)

        if self.use_bias:
            inputs = inputs + self.biases_2[block_index]

        if self.dropout is not None:
            inputs = self.dropout(inputs)

        inputs = inputs @ self.linear_layers_1[block_index]
        inputs = inputs * self.multipliers[block_index]
        if self.use_bias:
            inputs = inputs + self.biases_3[block_index]
        inputs = self.relu(inputs)

        return inputs + residual


class FeedForwardModel(pl.LightningModule):
    def __init__(self,
                 features_in: int,
                 class_count: int,
                 train_confidence: bool,
                 use_residual: bool,
                 depth: int = 2,
                 width: int = 64):
        super(FeedForwardModel, self).__init__()
        self.features_in = features_in
        self.class_count = class_count
        self.train_confidence = train_confidence
        self.use_residual = use_residual

        if train_confidence:
            class_count += 1

        layers = [nn.Linear(features_in, width), nn.ReLU()]
        # layers = []
        if use_residual:
            layers += [ResidualLinear(width, depth, dropout=0.0, use_bias=True, pre_activate=True)]
        else:
            for _ in range(depth):
                layers += [
                    nn.Linear(width, width),
                    nn.ReLU()
                ]
        layers += [nn.Linear(width, class_count, bias=False)]

        self.encoder = nn.Sequential(*layers)
        self.loss_module = nn.NLLLoss() if train_confidence else nn.CrossEntropyLoss()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        outputs = self.encoder(batch[0])
        if self.train_confidence:
            outputs = outputs[..., :-1]
        return torch.softmax(outputs, dim=-1)

    def training_step(self, batch, **kwargs) -> STEP_OUTPUT:
        inputs, labels = batch
        inputs += torch.randn_like(inputs) * 1e-1
        loss = self.compute_loss(inputs, labels)
        self.log("train_loss", loss.mean(), logger=False, on_step=True, prog_bar=True)

        if self.lr_schedulers() is not None:
            self.log("lr", self.lr_schedulers().get_lr()[0], on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_index, **kwargs) -> STEP_OUTPUT:
        inputs, labels = batch
        with torch.no_grad():
            loss = self.compute_loss(inputs, labels)
        self.log("validation_loss", loss.mean(), logger=False, on_step=False, prog_bar=True)
        return loss

    def compute_loss(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        from utils.tensor_utils import lerp

        logits: torch.Tensor = self.encoder(inputs)

        if self.train_confidence:
            confidence: torch.Tensor = logits[..., -1]
            logits = logits[..., :-1]
            probabilities = torch.softmax(logits, dim=-1)

            random_mask = torch.bernoulli(torch.full(confidence.size(), 0.5, device=confidence.device))
            masked_confidence = lerp(torch.sigmoid(confidence), 1.0, random_mask)

            class_count = logits.size(-1)
            one_hot_labels = nn.functional.one_hot(labels.to(torch.int64), class_count).to(logits.dtype)
            masked_confidence = masked_confidence.unsqueeze(-1).expand_as(one_hot_labels)
            probabilities = lerp(one_hot_labels, probabilities, masked_confidence)

            confidence_loss = torch.mean(torch.log(torch.exp(-confidence) + 1.0))
            base_loss = self.loss_module(torch.log(probabilities), labels.to(torch.int64))
            loss = base_loss + confidence_loss * 5e-1

            from torchmetrics import PearsonCorrCoef
            pearson = PearsonCorrCoef(num_outputs=class_count).to(logits.device)
            confidence_per_logit = confidence.unsqueeze(-1).tile([1, class_count])
            conf_corr = pearson(logits, confidence_per_logit)
            loss += torch.abs(conf_corr).mean() * 5e-2
        else:
            loss = self.loss_module(logits, labels.to(torch.int64))

        # loss = nn.functional.cross_entropy(outputs, labels)
        return loss

    def configure_optimizers(self) -> Any:
        # from utils.adams import AdamS
        # optimizer = AdamS(self.parameters(), lr=1e-3, weight_decay=1e-6)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # from utils.lr_schedule import CosineAnnealingDecayLR
        # scheduler = CosineAnnealingDecayLR(optimizer,
        #                                    warmup_epochs=0, max_epochs=50,
        #                                    decay=0.997, eta_min=2.5e-4)
        # return [optimizer], [scheduler]
        return optimizer


def train_and_predict_nn(train_inputs: np.ndarray,
                         train_labels: np.ndarray,
                         val_inputs: np.ndarray,
                         val_labels: np.ndarray,
                         test_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader, TensorDataset
    from pytorch_lightning.callbacks import EarlyStopping

    features_in = train_inputs.shape[-1]
    class_count = train_labels.max() + 1

    model = FeedForwardModel(features_in, class_count,
                             train_confidence=False,
                             use_residual=False,
                             depth=2,
                             width=64)

    train_dataset = TensorDataset(torch.as_tensor(train_inputs),
                                  torch.as_tensor(train_labels, dtype=torch.int64))
    train_loader = DataLoader(train_dataset, batch_size=train_inputs.shape[0], shuffle=True)
    train_loader.batch_sampler.sampler.datasets = train_dataset

    val_dataset = TensorDataset(torch.as_tensor(val_inputs),
                                torch.as_tensor(val_labels, dtype=torch.int64))
    val_loader = DataLoader(val_dataset, batch_size=val_inputs.shape[0], shuffle=False)
    val_loader.batch_sampler.sampler.datasets = val_dataset

    callbacks = [EarlyStopping(monitor="validation_loss", patience=10, verbose=False)]
    trainer = pl.Trainer(max_epochs=200,
                         accelerator="cpu",
                         enable_model_summary=False,
                         detect_anomaly=False,
                         logger=False,
                         callbacks=callbacks,
                         enable_checkpointing=False,
                         num_sanity_val_steps=0,
                         )
    trainer.fit(model, train_loader, val_loader)

    # region Predict
    train_dataset = TensorDataset(torch.as_tensor(train_inputs))
    train_loader = DataLoader(train_dataset, batch_size=train_inputs.shape[0], shuffle=False)
    train_loader.batch_sampler.sampler.datasets = train_dataset
    train_proba = trainer.predict(model, train_loader)
    train_proba = torch.concat(train_proba, dim=0).numpy()

    val_dataset = TensorDataset(torch.as_tensor(val_inputs))
    val_loader = DataLoader(val_dataset, batch_size=val_inputs.shape[0], shuffle=False)
    val_loader.batch_sampler.sampler.datasets = val_dataset
    val_proba = trainer.predict(model, val_loader)
    val_proba = torch.concat(val_proba, dim=0).numpy()

    test_dataset = TensorDataset(torch.as_tensor(test_inputs))
    test_loader = DataLoader(test_dataset, batch_size=val_inputs.shape[0], shuffle=False)
    test_loader.batch_sampler.sampler.datasets = test_dataset
    test_proba = trainer.predict(model, test_loader)
    test_proba = torch.concat(test_proba, dim=0).numpy()
    # endregion

    return train_proba, val_proba, test_proba


# endregion

def train_and_predict(train_inputs: np.ndarray,
                      train_labels: np.ndarray,
                      val_inputs: np.ndarray,
                      val_labels: np.ndarray,
                      test_inputs: np.ndarray,
                      method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "SVM":
        model = svm.SVC(probability=True)
        model.fit(train_inputs, train_labels)
        train_proba = model.predict_proba(train_inputs)
        val_proba = model.predict_proba(val_inputs)
        test_proba = model.predict_proba(test_inputs)

    elif method == "OCSVM":
        model = svm.OneClassSVM(nu=0.1)

        negative_train_inputs = train_inputs[train_labels == 0]
        model.fit(negative_train_inputs)
        train_score = - model.decision_function(train_inputs)
        val_score = - model.decision_function(val_inputs)
        test_score = - model.decision_function(test_inputs)

        # train_min_score, train_max_score = train_score.min(), train_score.max()
        val_min_score, val_max_score = val_score.min(), val_score.max()

        def normalize_score(score):
            score = (score - val_min_score) / (val_max_score - val_min_score)
            score = np.clip(score, 0.0, 1.0)
            return score

        train_proba = normalize_score(train_score)
        val_proba = normalize_score(val_score)
        test_proba = normalize_score(test_score)

    elif method == "RandomForest":
        model = ensemble.RandomForestClassifier()
        model.fit(train_inputs, train_labels)
        train_proba = model.predict_proba(train_inputs)
        val_proba = model.predict_proba(val_inputs)
        test_proba = model.predict_proba(test_inputs)

    elif method == "LinearRegression":
        model = linear_model.LinearRegression()
        model.fit(train_inputs, train_labels)
        train_proba = model.predict(train_inputs)
        val_proba = model.predict(val_inputs)
        test_proba = model.predict(test_inputs)

    elif method == "LogisticRegression":
        model = linear_model.LogisticRegression(max_iter=1000)
        model.fit(train_inputs, train_labels)
        train_proba = model.predict_proba(train_inputs)
        val_proba = model.predict_proba(val_inputs)
        test_proba = model.predict_proba(test_inputs)

    elif method == "NN":
        train_proba, val_proba, test_proba = train_and_predict_nn(train_inputs, train_labels,
                                                                  val_inputs, val_labels,
                                                                  test_inputs)
    else:
        raise RuntimeError()

    del model
    return train_proba, val_proba, test_proba


def remove_samples_without_features(fold: PresetFold) -> PresetFold:
    updated_samples = {subset_id: [] for subset_id in fold.samples}

    for subset_id, subset_samples in fold.samples.items():
        for sample in subset_samples:
            if sample.has_scalar_features or sample.has_categorical_features:
                updated_samples[subset_id].append(sample)

    fold.samples = updated_samples
    return fold


def get_inputs(fold: PresetFold) -> dict[SubsetID, dict[str, np.ndarray]]:
    if fold.has_scalar_features:
        scalar_preprocessor = ScalarPreprocess(use_standardization=True)
        scalar_preprocessor.fit(fold.samples[SubsetID.TRAIN])
    else:
        scalar_preprocessor = None

    if fold.has_categorical_features:
        categorical_preprocessor = CategoricalPreprocess(one_hot=True, categories=fold.categories_names)
        categorical_preprocessor.fit(fold.samples[SubsetID.TRAIN])
    else:
        categorical_preprocessor = None

    inputs = {}
    for subset_id, subset_samples in fold.samples.items():
        subset_inputs = {}
        for sample in subset_samples:
            data = []
            if fold.has_scalar_features:
                data.append(scalar_preprocessor(sample.scalar_features))
            if fold.has_categorical_features:
                data.append(categorical_preprocessor(sample.categorical_features))

            if len(data) == 1:
                data = data[0]
            elif len(data) == 2:
                data = np.concatenate(data)
            else:
                raise RuntimeError(len(data))
            subset_inputs[sample.id] = data
        inputs[subset_id] = subset_inputs
    return inputs


def get_normalized_inputs(inputs: dict[str, np.ndarray]) -> np.ndarray:
    inputs = np.asarray(list(inputs.values()))
    # inputs = inputs / np.linalg.norm(inputs, axis=-1, keepdims=True)
    return inputs


def project_to_positive_class(probabilities: torch.Tensor,
                              labels: torch.Tensor,
                              positive_class: int
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    if positive_class < 0:
        if len(probabilities.shape) > 1:
            positive_class = probabilities.shape[-1] + positive_class
        else:
            positive_class = 1

    if len(probabilities.shape) > 1:
        probabilities = probabilities[..., positive_class]

    labels = (labels == positive_class).to(torch.int32)

    return probabilities, labels


def run_experiment_on_fold(fold: PresetFold,
                           method: str,
                           labels_ratio=None,
                           positive_class: int = None,
                           keep_featureless=False,
                           ) -> tuple[pd.DataFrame, dict[str, float]]:
    if not keep_featureless:
        fold = remove_samples_without_features(fold)

    if labels_ratio is not None:
        count = int(len(fold.samples[SubsetID.TRAIN]) * labels_ratio)
        fold.samples[SubsetID.TRAIN] = fold.samples[SubsetID.TRAIN][:count]

    # region Get data (inputs/labels)
    inputs = get_inputs(fold)
    labels = {subset_id: [sample.label for sample in subset_samples]
              for subset_id, subset_samples in fold.samples.items()}

    train_inputs = inputs[SubsetID.TRAIN]
    train_labels = labels[SubsetID.TRAIN]

    if SubsetID.VALIDATION in inputs:
        val_inputs = inputs[SubsetID.VALIDATION]
        val_labels = labels[SubsetID.VALIDATION]
    else:
        indices = np.random.permutation(len(train_inputs))
        keys = list(train_inputs.keys())
        keys = [keys[i] for i in indices]
        val_count = int(len(train_inputs) * 0.2)
        val_keys = keys[:val_count]
        train_keys = keys[val_count:]
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]

        val_inputs = {key: train_inputs[key] for key in val_keys}
        train_inputs = {key: train_inputs[key] for key in train_keys}

        train_labels = np.asarray(train_labels)
        val_labels = train_labels[val_indices].tolist()
        train_labels = train_labels[train_indices].tolist()

    test_inputs = inputs[SubsetID.TEST]
    test_labels = labels[SubsetID.TEST]

    train_ids = list(train_inputs.keys())
    val_ids = list(val_inputs.keys())
    test_ids = list(test_inputs.keys())

    train_inputs = get_normalized_inputs(train_inputs)
    val_inputs = get_normalized_inputs(val_inputs)
    test_inputs = get_normalized_inputs(test_inputs)

    train_labels = np.asarray(train_labels, dtype=np.int32)
    val_labels = np.asarray(val_labels, dtype=np.int32)
    test_labels = np.asarray(test_labels, dtype=np.int32)
    # endregion

    train_proba, val_proba, test_proba = train_and_predict(train_inputs, train_labels,
                                                           val_inputs, val_labels,
                                                           test_inputs,
                                                           method)
    train_proba = torch.as_tensor(train_proba, dtype=torch.float32)
    train_labels = torch.as_tensor(train_labels, dtype=torch.float32)
    val_proba = torch.as_tensor(val_proba, dtype=torch.float32)
    val_labels = torch.as_tensor(val_labels, dtype=torch.int32)
    test_proba = torch.as_tensor(test_proba, dtype=torch.float32)
    test_labels = torch.as_tensor(test_labels, dtype=torch.int32)

    if positive_class is not None:
        train_proba, train_labels = project_to_positive_class(train_proba, train_labels, positive_class)
        val_proba, val_labels = project_to_positive_class(val_proba, val_labels, positive_class)
        test_proba, test_labels = project_to_positive_class(test_proba, test_labels, positive_class)

    # region Classification Summary
    metrics = [AUROC]
    if positive_class is not None:
        metrics += [
            EER,
            AccuracyAtEER,

            SensitivityAtEER,
            SpecificityAtEER,
        ]

        sensitivity_metrics = make_metrics_for_sensitivity_threshold(base_metrics=[
            SensitivityThreshold,
            AccuracyAtSensitivity,
            SensitivityAtSensitivity,
            SpecificityAtSensitivity
        ],
            relative_threshold=.90)
        sensitivity_threshold = sensitivity_metrics.pop(0)
        metrics += sensitivity_metrics
    else:
        metrics += [
            AccuracyAverage,
            SensitivityAverage,
            SpecificityAverage,
            F1ScoreAverage
        ]
        sensitivity_threshold = None
    sensitivity_threshold: type[SensitivityThreshold] | None

    classification_summary = ClassificationSummary(metrics)
    # passing probabilities as logits since we don't have logits
    results = classification_summary(test_proba, test_labels, probabilities=test_proba)
    results = {metric_name: float(metric_value) for metric_name, metric_value in results.items()}
    # endregion

    # region Panda dataframe from inferences (ids/predictions/labels/...)
    if positive_class is not None:
        classification_summary.eer_threshold = EERThreshold.compute_threshold(val_proba, val_labels)
        classification_summary.sensitivity_threshold = sensitivity_threshold.compute_threshold(val_proba, val_labels)

    train_inferences_data = classification_summary.get_available_inferences(train_proba, train_labels)
    train_inferences_index = pd.Index(train_ids, name="ScanID")
    train_inferences = pd.DataFrame(train_inferences_data, index=train_inferences_index)
    train_inferences["SubsetID"] = "train"

    val_inferences_data = classification_summary.get_available_inferences(val_proba, val_labels)
    val_inferences_index = pd.Index(val_ids, name="ScanID")
    val_inferences = pd.DataFrame(val_inferences_data, index=val_inferences_index)
    val_inferences["SubsetID"] = "validation"

    test_inferences_data = classification_summary.get_available_inferences(test_proba, test_labels)
    test_inferences_index = pd.Index(test_ids, name="ScanID")
    test_inferences = pd.DataFrame(test_inferences_data, index=test_inferences_index)
    test_inferences["SubsetID"] = "test"

    inferences = pd.concat([test_inferences, train_inferences, val_inferences])
    # endregion

    return inferences, results


def run_experiment_on_dataset(folds: list[PresetFold],
                              method: str,
                              labels_ratio=None,
                              positive_class: int = None,
                              keep_featureless=False,
                              seeds: list[int] | tuple[int, ...] = (3930787959, 3048086154, 1308332165,
                                                                    4198623908, 2751205564),
                              ) -> tuple[list[pd.DataFrame], list[dict[str, float]]]:
    folds_inferences: list[pd.DataFrame] = []
    folds_metrics: list[dict[str, float]] = []
    for i, fold in enumerate(folds):
        print("-> Running fold n°{}".format(i))
        seed_index = i % len(seeds)
        set_determinism(seeds[seed_index])
        seed_everything(seeds[seed_index])

        fold_inferences, fold_metrics = run_experiment_on_fold(fold, method, labels_ratio,
                                                               positive_class, keep_featureless)
        fold_inferences["Fold"] = i
        folds_inferences.append(fold_inferences)
        folds_metrics.append(fold_metrics)

    return folds_inferences, folds_metrics


def get_dataset_folds(folder_path: Path,
                      fold_path_pattern: str = "*fold_*.csv",
                      scalar_path_pattern: str = "scalar_features.csv",
                      categorical_path_pattern: str = "categorical_features.csv",
                      ) -> list[PresetFold] | None:
    scalars_path = None
    categorical_path = None
    folds_paths = []

    for filepath in folder_path.iterdir():
        if filepath.match(fold_path_pattern):
            folds_paths.append(filepath)

        elif filepath.match(scalar_path_pattern):
            scalars_path = filepath

        elif filepath.match(categorical_path_pattern):
            categorical_path = filepath

    if (scalars_path is None) and (categorical_path is None):
        return None

    folds = [PresetFold(filepath, scalar_features_path=scalars_path,
                        categorical_features_path=categorical_path)
             for filepath in folds_paths]

    return folds


def get_dataset_name(experiment_name: str) -> str:
    return experiment_name.split("-")[0]


def make_inferences_folders(root: Path) -> tuple[Path, Path]:
    inferences_folder = root / "inferences"
    csv_folder = inferences_folder / "csv"
    excel_folder = inferences_folder / "excel"

    if not csv_folder.exists():
        csv_folder.mkdir(parents=True)

    if not excel_folder.exists():
        excel_folder.mkdir(parents=True)

    return csv_folder, excel_folder


def save_experiment_inferences(folds_inferences: list[pd.DataFrame],
                               experiment_name: str,
                               csv_folder: Path,
                               excel_folder: Path,
                               ) -> None:
    experiment_folder = csv_folder / experiment_name
    experiment_folder.mkdir(parents=True, exist_ok=True)

    for i, fold_inferences in enumerate(folds_inferences):
        fold_inferences.to_csv(experiment_folder / "inferences_fold_{:02d}.csv".format(i))

    experiment_inferences = pd.concat(folds_inferences)
    experiment_inferences.to_csv(csv_folder / "inferences_{}.csv".format(experiment_name))
    experiment_inferences.to_excel(excel_folder / "inferences_{}.xlsx".format(experiment_name))


def main():
    import logging
    logging.disable(logging.INFO)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--folder", required=True, type=str)
    arg_parser.add_argument("--methods", default=["LogisticRegression"], nargs="+")

    arg_parser.add_argument("--positive_class", default=None, type=int)
    arg_parser.add_argument("--labels_ratios", nargs="+")
    arg_parser.add_argument("--keep_featureless", action="store_true")

    arg_parser.add_argument("--fold_pattern", default="*fold_*.csv")
    arg_parser.add_argument("--scalar_pattern", default="scalar_features.csv")
    arg_parser.add_argument("--categorical_pattern", default="categorical_features.csv")

    arg_parser.add_argument("--seeds", nargs="+",
                            default=[3930787959, 3048086154, 1308332165,
                                     4198623908, 2751205564])

    args = arg_parser.parse_args()
    folder = Path(args.folder)
    labels_ratios: list[float | None] = ([float(x) for x in args.labels_ratios]
                                         if args.labels_ratios is not None else None)
    positive_class = int(args.positive_class) if args.positive_class is not None else None
    path_patterns = {
        "fold_path_pattern": args.fold_pattern,
        "scalar_path_pattern": args.scalar_pattern,
        "categorical_path_pattern": args.categorical_pattern,
    }
    seeds = args.seeds

    sub_folders = {sub_folder: get_dataset_folds(sub_folder, **path_patterns)
                   for sub_folder in folder.iterdir() if sub_folder.is_dir()}
    if labels_ratios is None:
        labels_ratios = [None]

    experiments = {(sub_folder, ratio): folds
                   for sub_folder, folds in sub_folders.items()
                   for ratio in labels_ratios
                   if folds is not None}

    csv_folder, excel_folder = make_inferences_folders(folder)

    # all_stats = None
    # methods = ["NN", "SVM", "RandomForest", "LogisticRegression"]
    datasets_metrics: dict[str, dict[str, float]] = {}
    folds_indices: dict[str, list[str]] = {}
    methods = args.methods
    keep_featureless = args.keep_featureless
    for (sub_folder, labels_ratio), folds in experiments.items():
        for method in methods:
            # region Experiment name
            if len(methods) > 1:
                experiment_name = "{}_{}".format(sub_folder.stem, method)
            else:
                experiment_name = sub_folder.stem
            if labels_ratio is not None:
                labels_ratio: float
                experiment_name = "{}_{:02d}".format(experiment_name, int(labels_ratio * 100))
            # endregion

            folds = get_dataset_folds(sub_folder, **path_patterns)
            if folds is None:
                continue

            print("Running experiment on {}".format(experiment_name))
            folds_inferences, dataset_metrics = run_experiment_on_dataset(folds, method,
                                                                          labels_ratio, positive_class,
                                                                          keep_featureless,
                                                                          seeds)
            save_experiment_inferences(folds_inferences, experiment_name, csv_folder, excel_folder)

            folds_names = []
            for i, fold_metrics in enumerate(dataset_metrics):
                fold_name = "{}/version_{}".format(experiment_name, i)
                datasets_metrics[fold_name] = fold_metrics
                folds_names.append(fold_name)
            folds_indices[experiment_name] = folds_names

            # metrics_names = list(experiment_summary.keys())
            # metrics_stats = [experiment_summary[metric_name] for metric_name in metrics_names]
            # if all_stats is None:
            #     all_stats = pd.DataFrame(data=[metrics_stats], index=[experiment_name], columns=metrics_names)
            # else:
            #     all_stats.loc[experiment_name] = metrics_stats

    gathered_results = pd.DataFrame.from_dict(datasets_metrics, orient="index")
    gathered_results.index.name = "Test ID"
    gathered_results.to_csv(folder / "gathered_results.csv")

    experiments_summary = summarize_experiments(gathered_results, folds_indices.values())
    experiments_summary = sort_experiments_summary_columns(experiments_summary)
    experiments_summary.index = folds_indices.keys()
    experiments_summary["Method"] = methods * len(experiments)
    experiments_summary["Dataset"] = list(map(get_dataset_name, folds_indices))
    experiments_summary.to_csv(folder / "experiments_summary.csv")

    formatted_summary = format_experiments_summary(experiments_summary)
    formatted_summary.to_csv(folder / "formatted_summary.csv", index=False)
    # all_stats["Method"] = methods * len(experiments)
    # all_stats["Dataset"] = all_stats.index.map(lambda exp_id: exp_id.split("-")[0])
    # all_stats.to_csv(folder / "summary.csv")
    # all_stats.to_excel(folder / "summary.xlsx")
    # print(all_stats)
    # stats_means.to_csv(folder / "summary_means_only.csv")


if __name__ == "__main__":
    main()
