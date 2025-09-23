import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from abc import abstractmethod
from typing import Any

from data.transforms.serializable_transform import SerializableTransform


class BatchTransform(SerializableTransform):
    @abstractmethod
    def on_train_batch_start(self, module: nn.Module, batch: Any, batch_idx: int):
        pass

    @abstractmethod
    def on_train_batch_end(self, module: nn.Module, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        pass
