import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler
import pandas as pd
from typing import Union

from mindful_core.utils.data_constants import LABEL
from mindful_core.data.subset import Subset
from mindful_core.data.transforms.pipeline import Pipeline
from mindful_core.models.model_output import ClassifierOutput
from mindful_core.models.classification.abstract_classifier import AbstractClassifier


class DifficultySampler(Sampler):
    def __init__(self,
                 dataset: Subset,
                 trainer: pl.Trainer,
                 model: AbstractClassifier,
                 batch_size: int,
                 indices: list = None,
                 num_samples: int = None,
                 num_workers: int = 0
                 ) -> None:
        super(DifficultySampler, self).__init__(dataset)
        self.dataset = dataset
        self.trainer = trainer
        self.model = model
        self.class_count = dataset.get_class_count()
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        starting_history = 1 / self.class_count
        starting_history = torch.ones([len(dataset), self.class_count], dtype=torch.float32) * starting_history
        self.history: list[torch.Tensor] = [starting_history]

        df = pd.DataFrame(data={LABEL: self.get_labels()}, index=self.indices)
        df = df.sort_index()
        label_to_count = df[LABEL].value_counts()
        weights = 1.0 / label_to_count[df[LABEL]]
        self.weights = torch.DoubleTensor(weights.to_list())

        update_sampler = SequentialSampler(dataset)
        update_sampler.dataset = dataset
        self.update_data_loader = DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             sampler=update_sampler,
                                             persistent_workers=num_workers > 0,
                                             collate_fn=None)

        self.update_handle = self.model.register_train_epoch_end_hook(self.update)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))
        # (self.indices[i] for i in torch.arange(len(self.indices)))

    def __len__(self):
        return self.num_samples

    # noinspection PyUnusedLocal
    def update(self, *args, **kwargs) -> None:
        # region Get Predictions/Labels
        predictions = self.get_predictions()
        labels = torch.as_tensor(self.get_labels(), dtype=torch.int32)
        # endregion

        sample_count = predictions.shape[0]

        # region Filter by matching labels
        possible_labels = torch.arange(self.class_count, dtype=torch.int32)
        match_labels = possible_labels.unsqueeze(0) == labels.unsqueeze(1)
        not_labels = ~match_labels

        previous_predictions = self.history[-1]
        predictions_delta = predictions - previous_predictions
        predictions_ratio = torch.log(predictions / previous_predictions)

        predictions_delta_right_label: torch.Tensor = predictions_delta[match_labels]
        predictions_delta_other_label: torch.Tensor = predictions_delta[not_labels]
        predictions_delta_other_label = predictions_delta_other_label.reshape([sample_count, self.class_count - 1])

        predictions_ratio_right_label: torch.Tensor = predictions_ratio[match_labels]
        predictions_ratio_other_label: torch.Tensor = predictions_ratio[not_labels]
        predictions_ratio_other_label = predictions_ratio_other_label.reshape(predictions_delta_other_label.shape)
        # endregion

        unlearning_difficulty = ((predictions_delta_right_label.clamp(min=0.0) *
                                  predictions_ratio_right_label) +
                                 (predictions_delta_other_label.clamp(max=0.0) *
                                  predictions_ratio_other_label).sum(dim=1))

        learning_difficulty = ((predictions_delta_right_label.clamp(max=0.0) *
                                predictions_ratio_right_label) +
                               (predictions_delta_other_label.clamp(min=0.0) *
                                predictions_ratio_other_label).sum(dim=1))

        difficulty: torch.Tensor = unlearning_difficulty + learning_difficulty
        total_difficulty = difficulty.sum()
        new_weights = difficulty / (total_difficulty + 1e-5)
        self.weights = new_weights
        self.history.append(predictions)

    def get_predictions(self):
        with Pipeline.no_labels(), self.dataset.pipeline.no_augmentation():
            was_training = self.model.training
            if was_training:
                self.model.eval()

            with torch.no_grad():
                outputs: list[ClassifierOutput] = [self.model(self.to_device(batch))
                                                   for batch in self.update_data_loader]

            if was_training:
                self.model.train()

        predictions = ClassifierOutput.concat_logits(outputs)
        if len(predictions.shape) == 1:
            predictions = torch.sigmoid(predictions)
            predictions = torch.stack([1.0 - predictions, predictions], dim=-1)
        else:
            predictions = torch.softmax(predictions, dim=1)

        predictions = predictions.cpu()
        return predictions

    def to_device(self,
                  data: Union[tuple[torch.Tensor, ...], list[torch.Tensor], torch.Tensor]
                  ) -> Union[tuple[torch.Tensor, ...], torch.Tensor]:
        if isinstance(data, (tuple, list)):
            data = tuple([x.to(self.model.device) for x in data])
        else:
            data = data.to(self.model.device)
        return data

    def get_labels(self) -> list[int]:
        return [self.dataset.samples[i].label for i in self.indices]
