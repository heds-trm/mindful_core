import numpy as np
import torch
# noinspection PyPackageRequirements
import umap
from typing import Optional, Sequence, Union

from mindful_core.data import SubsetID
from mindful_core.models.representation.callbacks.representation_model_callback import RepresentationModelCallback


class SeparabilityLogger(RepresentationModelCallback):
    def __init__(self,
                 frequency: int,
                 n_components: int,
                 n_iterations: int,
                 seed: Optional[int]):
        super(SeparabilityLogger, self).__init__(frequency=frequency,
                                                 used_subsets=[SubsetID.VALIDATION])
        self.frequency = frequency
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.seed = seed

    def apply_umap(self, representations: np.ndarray):
        projector = umap.UMAP(n_components=self.n_components, n_epochs=self.n_iterations, random_state=self.seed)
        return projector.fit_transform(representations)

    def _on_representation_end(self,
                               representations: torch.Tensor,
                               subset: SubsetID,
                               metadata: Union[Sequence[str], list[Sequence[str]]] = None,  # labels
                               metadata_header: Sequence[str] = None,  # labels header
                               global_step: int = None):
        # representations: np.ndarray = representations.to("cpu").numpy()
        # representations = self.apply_umap(representations)
        # kmeans = KMeans(n_clusters=2, max_iter=self.n_iterations, random_state=self.seed)
        # distance_to_clusters = kmeans.fit_transform(representations)

        raise NotImplementedError
