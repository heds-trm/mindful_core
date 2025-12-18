import torch
from abc import abstractmethod, ABC

class AutoencoderInterface(ABC):
    @abstractmethod
    def autoencode(self, inputs: torch.Tensor | tuple[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor]:
        raise NotImplementedError("`autoencode` must be implemented in subclasses.")
    