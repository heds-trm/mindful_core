import torch
from typing import Sequence, Union

from mindful_core.data.subset_id import SubsetID
from mindful_core.models.representation.callbacks import RepresentationModelCallback


class Projector(RepresentationModelCallback):
    def __init__(self, frequency: int, max_count: int):
        super(Projector, self).__init__(frequency=frequency,
                                        used_subsets=[SubsetID.VALIDATION, SubsetID.TEST])
        self.max_count = max_count

    def _on_representation_end(self,
                               representations: torch.Tensor,
                               subset: SubsetID,
                               metadata: Union[Sequence[str], list[Sequence[str]]] = None,  # labels
                               metadata_header: Sequence[str] = None,  # labels header
                               global_step: int = None):
        if self.logger is None:
            if self.logger_fn is not None:
                self.logger = self.logger_fn()
            if self.logger is None:
                return

        if self.max_count is not None and representations.size(0) > self.max_count:
            indices = torch.randperm(representations.size(0))
            representations = representations[indices][:self.max_count]

        if len(representations.shape) > 2:
            representations = torch.flatten(representations, start_dim=1)

        tag = "{}_representations".format(subset.as_prefix())
        # noinspection PyTypeChecker
        self.logger.experiment.add_embedding(representations,
                                             metadata=metadata, metadata_header=metadata_header,
                                             global_step=global_step, tag=tag)
