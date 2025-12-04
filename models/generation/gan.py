from models.generation.abstract_generator import AbstractGenerator


class GAN(AbstractGenerator):
    def __init__(self,
                 optimizer_config: dict[str, dict[str, Any]] | None = None,
                 **kwargs
                 ):
        super().__init__(optimizer_config, **kwargs)
