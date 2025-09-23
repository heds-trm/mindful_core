import numpy as np
import torch
from torch.utils import data
from pathlib import Path

from data import Sample
from data.transforms.pipeline import Pipeline


class Subset(data.Dataset):
    def __init__(self,
                 samples: list[Sample],
                 pipeline: Pipeline | Path | str,
                 verbose=True,
                 **kwargs
                 ):
        if isinstance(pipeline, (str, Path)):
            pipeline = Pipeline(pipeline, multiview=kwargs.get("multiview", False))

        self.samples: list[Sample] = samples
        self.pipeline: Pipeline = pipeline
        self.verbose: bool = verbose

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> torch.Tensor | tuple[torch.Tensor, ...]:
        self._check_sample_index(index)
        index = index % len(self.samples)
        return self.pipeline(self.samples[index])

    # region Get sample
    def get_labels(self) -> list[int]:
        return [sample.label for sample in self.samples]

    def get_class_count(self) -> int:
        return len(np.unique(self.get_labels()))

    def _check_sample_index(self, index: int) -> None:
        if (index >= len(self.samples)) or (index < 0):
            print("===================" * 4)
            print(index, len(self.samples))
            print("===================" * 4)
            # raise ValueError("Index `{}` is outside the expected range [0 , {}].".
            #                  format(index, len(self.samples) - 1))

    # endregion
