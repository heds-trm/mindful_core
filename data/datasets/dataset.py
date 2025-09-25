import numpy as np
from torch.utils.data import DataLoader, Sampler, SequentialSampler
import pytorch_lightning as pl
from pathlib import Path
from typing import Iterator, Sized
import os

from mindful_core.data import Subset, SubsetID, Sample
from mindful_core.data.data_folds import DataFold, PresetFold
from mindful_core.data.transforms.pipeline import Pipeline


class MindfulSampler(Sampler):
    def __init__(self, dataset: Sized, shuffle: bool):
        super(MindfulSampler, self).__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle
        self.random_state = np.random.RandomState()

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            yield from self.random_state.permutation(len(self.dataset))
        else:
            yield from range(len(self.dataset))

    def __len__(self):
        return len(self.dataset)


class MindfulDataset(object):
    def __init__(self,
                 fold: str | Path | DataFold,
                 seed: int = None,
                 training_data_reduction: float = None
                 ):
        self.fold: DataFold = fold if isinstance(fold, DataFold) else PresetFold(fold, seed=seed)
        self.random_state = np.random.RandomState(seed)
        self.training_data_reduction = training_data_reduction
        self.samples: dict[SubsetID, list[Sample]] = self.fold.samples

        if self.should_reduce_training_data:
            self.samples = self.reduce_training_data()

        for subset_id, subset_samples in self.samples.items():
            labels_ratios = DataFold.get_labels_ratios(subset_samples)
            labels_ratios = [round(ratio, 3) for ratio in labels_ratios]
            print("Subset `{}` contains {} samples with the following label ratios: {}."
                  .format(subset_id, len(subset_samples), labels_ratios))

    def reduce_training_data(self, samples: dict[SubsetID, list[Sample]] = None):
        samples = self.samples if samples is None else samples
        if (self.training_data_reduction is None) or (SubsetID.TRAIN not in samples):
            return samples

        training_samples = samples[SubsetID.TRAIN]
        # noinspection PyTypeChecker
        self.random_state.shuffle(training_samples)

        if self.training_data_reduction > 1:
            new_training_samples_count = int(self.training_data_reduction)
        else:
            new_training_samples_count = int(np.ceil(len(training_samples) * self.training_data_reduction))

        samples[SubsetID.TRAIN] = training_samples[:new_training_samples_count]
        return samples

    def get_data_loaders(self,
                         pipelines: dict[SubsetID, Pipeline],
                         batch_sizes: dict[SubsetID, int | str],
                         num_workers: int,
                         use_imbalanced_sampler: bool = False,
                         trainer: pl.Trainer = None,
                         model=None,
                         training: bool = True,
                         ) -> dict[SubsetID, DataLoader]:
        data_loaders = {}
        fit_samples = self.get_fit_samples()

        for subset_id in pipelines:
            pipelines[subset_id].fit(fit_samples)

        for subset_id, subset_samples in self.samples.items():
            subset = Subset(samples=subset_samples, pipeline=pipelines[subset_id])
            shuffle = (subset_id == SubsetID.TRAIN) and training

            batch_size = batch_sizes[subset_id]
            if isinstance(batch_size, str):
                if batch_size == "full_batch":
                    batch_size = len(subset)
                else:
                    raise ValueError("When `batch_size` is a string, it is expected to be in {'full_batch'}.")

            if use_imbalanced_sampler:
                if shuffle:
                    if use_imbalanced_sampler == "v2":
                        from mindful_core.data.sampling.difficulty_sampler import DifficultySampler
                        sampler = DifficultySampler(subset, trainer, model, batch_size)
                    else:
                        from mindful_core.data.sampling.torchsampler_imbalanced import ImbalancedDatasetSampler
                        sampler = ImbalancedDatasetSampler(subset)
                else:
                    sampler = SequentialSampler(subset)
                # Bugfix: The data loader will be looking for `sampler.dataset` which is under `sampler.data_source`
                sampler.dataset = subset
            else:
                sampler = MindfulSampler(subset, shuffle)

            subset_workers = num_workers if subset_id != SubsetID.VALIDATION else 0
            data_loader = DataLoader(subset,
                                     batch_size=batch_size,
                                     num_workers=subset_workers,
                                     sampler=sampler,
                                     persistent_workers=subset_workers > 0,
                                     collate_fn=None)
            data_loaders[subset_id] = data_loader
        return data_loaders

    def get_fit_samples(self) -> list[Sample]:
        if self.has_samples_for_subset(SubsetID.TRAIN):
            return self.samples[SubsetID.TRAIN]
        elif self.has_samples_for_subset(SubsetID.VALIDATION):
            return self.samples[SubsetID.VALIDATION]
        elif self.has_samples_for_subset(SubsetID.TEST):
            return self.samples[SubsetID.TEST]
        else:
            raise RuntimeError("No samples were found for any subset.")

    def has_samples_for_subset(self, subset_id: SubsetID) -> bool:
        return (subset_id in self.samples) and (len(self.samples[subset_id]) > 0)

    def get_batch_samples_iterators(self,
                                    batch_sizes: dict[SubsetID, int]
                                    ) -> dict[SubsetID, Iterator[list[Sample]]]:
        # noinspection PyTypeChecker
        return {subset_id: MindfulSampleBatchIterator(self.samples[subset_id], batch_sizes[subset_id])
                for subset_id in self.samples}

    def save_metadata(self, dir_path: str):
        # TODO: Provide labels to representation logger
        for subset_id, subset in self.samples.items():
            subset_name = str(subset_id).lower()
            metadata_path = os.path.join(dir_path, "{}.tsv".format(subset_name))
            with open(metadata_path, "w") as metadata_file:
                metadata_file.write(Sample.get_tsv_header(subset))
                for sample in subset:
                    metadata_file.write(sample.to_tsv_line())

    @property
    def should_reduce_training_data(self) -> bool:
        return self.training_data_reduction is not None


class MindfulSampleBatchIterator(object):
    def __init__(self, samples: list[Sample], batch_size: int) -> None:
        self.samples = samples
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[Sample]]:
        batch = [None] * self.batch_size
        idx_in_batch = 0
        for sample in self.samples:
            # noinspection PyTypeChecker
            batch[idx_in_batch] = sample
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                # noinspection PyTypeChecker
                yield batch
                idx_in_batch = 0
                batch = [None] * self.batch_size
        if idx_in_batch > 0:
            # noinspection PyTypeChecker
            yield batch[:idx_in_batch]
