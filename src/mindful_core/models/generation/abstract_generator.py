from pytorch_lightning.utilities.types import STEP_OUTPUT
from abc import abstractmethod
from typing import Type, Self, Any

from mindful_core.models.module import MindfulModule, LossAggregator
from mindful_core.models.model_output import GeneratorOutput
from mindful_core.data.subset_id import SubsetID


class AbstractGenerator(MindfulModule):
    # region Class/Subclass methods
    @classmethod
    @abstractmethod
    def module_identifier(cls) -> str:
        raise NotImplementedError("Method `module_identifier` must be implemented in subclasses")

    @classmethod
    def get_registered_generators(cls) -> dict[str, Type[Self]]:
        return {key: value for key, value in cls._module_register if issubclass(value, AbstractGenerator)}

    @classmethod
    def base_log_monitors(cls) -> list[str]:
        return ["loss"]

    # endregion

    def __init__(self,
                 optimizer_config: dict[str, dict[str, Any]] | None = None,
                 **kwargs
                 ):
        super().__init__(optimizer_config, **kwargs)

    # region Forward
    def forward(self, inputs, *args, **kwargs) -> GeneratorOutput:
        raise NotImplementedError

    def generate(self, *args, **kwargs) -> GeneratorOutput:
        raise NotImplementedError
    
    # endregion

    def base_step(self, batch, subset_id: SubsetID, **model_kwargs) -> STEP_OUTPUT:
        raise NotImplementedError
