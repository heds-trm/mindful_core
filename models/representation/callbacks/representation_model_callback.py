import torch
from pytorch_lightning.loggers import TensorBoardLogger
from abc import abstractmethod
from typing import Union, Sequence, Optional, Callable

from mindful_core.data.subset_id import SubsetID

RepresentationTensors = Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]]


class RepresentationModelCallback(object):
    def __init__(self, frequency: int, used_subsets: list[SubsetID]):
        self.frequency = frequency
        self.used_subsets = used_subsets
        self.logger: Optional[TensorBoardLogger] = None
        self.logger_fn: Optional[Callable[[], TensorBoardLogger]] = None

    def should_run(self, global_step: int = None):
        if global_step is None:
            return True

        if global_step == 0:
            return True

        return ((global_step + 1) % self.frequency) == 0

    def uses_subset(self, subset_id: SubsetID):
        return subset_id in self.used_subsets

    @staticmethod
    def flatten_representation(representations):
        if len(representations.shape) > 2:
            representations = torch.flatten(representations, start_dim=1)
        return representations

    @abstractmethod
    def _on_representation_end(self,
                               representations: torch.Tensor,
                               subset: SubsetID,
                               metadata: Union[Sequence[str], list[Sequence[str]]] = None,  # labels
                               metadata_header: Sequence[str] = None,  # labels header
                               global_step: int = None):
        raise NotImplementedError

    def on_representation_end(self,
                              representations: RepresentationTensors,
                              subset: SubsetID,
                              metadata: Union[Sequence[str], list[Sequence[str]]] = None,  # labels
                              metadata_header: Sequence[str] = None,  # labels header
                              global_step: int = None):
        if self.should_run(global_step) and self.uses_subset(subset):
            if isinstance(representations, (tuple, list)):
                representations = torch.cat(representations, dim=0)

            representations = self.flatten_representation(representations)

            self._on_representation_end(representations=representations,
                                        subset=subset,
                                        metadata=metadata,
                                        metadata_header=metadata_header,
                                        global_step=global_step)
