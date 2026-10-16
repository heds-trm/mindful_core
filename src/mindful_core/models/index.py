import torch
from pathlib import Path
from typing import Any, TypedDict, NotRequired, Type, TypeVar

from mindful_core.utils.misc import try_load_json
from mindful_core.utils.parsing import parse_checkpoint_path
from mindful_core.models.module import MindfulModule
from mindful_core.models.classification.abstract_classifier import AbstractClassifier
from mindful_core.models.representation.abstract_representation_model import AbstractRepresentationModel

_MM = TypeVar("_MM", bound=MindfulModule)


class ModuleConfig(TypedDict):
    class_name: str
    hparams: str | dict[str, Any]
    checkpoint: NotRequired[str | None]
    sub_modules: NotRequired[dict[str, "ModuleConfig"] | None]
    unfreeze: NotRequired[str | None]
    train: NotRequired[str | None]


def get_model(module_config: ModuleConfig,
              checkpoint: str | Path | None = None,
              device: str | None = None
              ) -> MindfulModule:
    # region Check configuration
    if not isinstance(module_config, dict):
        raise ValueError("Expected `module_config` to be a dictionary, got `{}`".format(type(module_config)))

    mandatory_keys = ["class_name", "hparams"]
    missing_keys = [key for key in mandatory_keys if key not in module_config]
    if len(missing_keys) > 0:
        raise KeyError("Keys `{}` are missing from module config.".format(missing_keys))
    # endregion

    model_class_name = module_config["class_name"]
    hparams = module_config["hparams"]
    if checkpoint is None:
        checkpoint = module_config.get("checkpoint")
    sub_modules_config = module_config.get("sub_modules")
    unfreeze = module_config.get("unfreeze", False)
    train = module_config.get("train", False)

    if (sub_modules_config is not None) and (len(sub_modules_config) > 0):
        sub_modules = {module_id: get_model(sub_module_config)
                       for module_id, sub_module_config in sub_modules_config.items()}
    else:
        sub_modules = {}

    model_class = MindfulModule.get_module_class(model_class_name)
    if isinstance(hparams, (Path, str)):
        hparams = try_load_json(hparams, "Mindful Model hparams")

    if len(sub_modules) > 0:
        hparams.update(sub_modules)

    if checkpoint is not None:
        print("Loading a {} from {}.".format(model_class.__name__, checkpoint))
        if device is not None:
            map_location = torch.device(device)
        else:
            map_location = None
        # noinspection PyTypeChecker
        model = model_class.load_from_checkpoint(checkpoint, **hparams, map_location=map_location)
    else:
        print("Building new {}.".format(model_class.__name__))
        model = model_class(**hparams)
        if device is not None:
            model = model.to(device)

    if unfreeze:
        model.unfreeze()

    if train:
        model.train()

    return model


def get_models(modules_configs: dict[str, ModuleConfig | str],
               device: str | None = None) -> dict[str, MindfulModule]:
    """
    Builds models for each (`module_id`, `module_config`) pair in modules_configs.

    `module_config` can be a string referencing another module_id. If so, it will reference the same module,
    thus sharing weights.
    """
    original_order = list(modules_configs.keys())

    modules = {
        module_id: get_model(module_config, device=device)
        for module_id, module_config in modules_configs.items()
        if isinstance(module_config, dict)
    }
    
    reorder = False
    for module_id, module_config in modules_configs.items():
        if isinstance(module_config, str):
            if module_config not in modules_configs:
                raise RuntimeError("Misconfiguration error: could not find module for id `{}` in ({}).".
                                   format(module_config, list(modules_configs.keys())))
            modules[module_id] = modules[module_config]
            reorder = True
        elif not isinstance(module_config, dict):
            raise RuntimeError("Misconfiguration error: Unknown identifier type `{}` for id `{}`.".
                               format(type(module_config), module_id))
        
    if reorder:
        modules = {module_id: modules[module_id] for module_id in original_order}
        
    return modules


def is_model_of_type(module: str | Type[MindfulModule] | MindfulModule,
                     module_type: Type[MindfulModule] | tuple[Type[MindfulModule], ...]
                     ) -> bool:
    if isinstance(module, str):
        module = MindfulModule.get_module_class(module)

    if isinstance(module, MindfulModule):
        module = type(module)
    elif not isinstance(module, type):
        raise TypeError("Expected `module` to either be a string, a MindfulModule instance "
                        "or a MindfulModule class, got a `{}`".format(type(module)))

    return issubclass(module, module_type)


def is_classification_model(module: str | Type[MindfulModule] | MindfulModule) -> bool:
    return is_model_of_type(module, AbstractClassifier)


def is_representation_model(module: str | Type[MindfulModule] | MindfulModule) -> bool:
    return is_model_of_type(module, AbstractRepresentationModel)


# region Find and load models
def find_mindful_model_config(model_path: Path | str,
                              model_class: Type[MindfulModule]
                              ) -> Path | None:
    model_path = Path(model_path)
    config_paths = list(model_path.glob("*_config.json"))
    for model_class_name in model_class.get_registered_subclass_identifiers():
        for config_path in config_paths:
            if model_class_name not in config_path.stem:
                continue

            model_class = model_class.get_module_class(model_class_name)
            if not issubclass(model_class, model_class):
                continue

            return config_path
    return None


def find_mindful_models(models_paths: list[Path | str],
                        model_class: Type[MindfulModule]
                        ) -> list[Path]:
    models_paths = [models_paths] if isinstance(models_paths, (str, Path)) else models_paths
    models_paths = [Path(model_path) for model_path in models_paths]

    updated_models_paths: list[Path] = []
    for models_path in models_paths:
        if not models_path.is_dir():
            continue

        if (models_path / "lightning_logs").exists():
            models_path = models_path / "lightning_logs"

        if models_path.parent.name == "lightning_logs":
            # Single model case
            updated_models_paths.append(models_path)

        else:
            for model_path in models_path.iterdir():
                if model_path.is_dir():
                    config_path = find_mindful_model_config(model_path, model_class)
                    if config_path is not None:
                        updated_models_paths.append(model_path)

    return updated_models_paths


def load_mindful_model(model_path: Path | str,
                       model_class: Type[_MM],
                       monitor: str,
                       device: str | None = None,
                       load_weights: bool = True,
                       ) -> tuple[_MM | None, ModuleConfig | None]:
    model_path = Path(model_path)
    model_config_path = find_mindful_model_config(model_path, model_class)
    # noinspection PyTypeChecker
    model_config: ModuleConfig = try_load_json(model_config_path, file_description="Mindful Model config")

    if load_weights:
        checkpoint_path = parse_checkpoint_path(model_path / "checkpoints", monitor)
    else:
        checkpoint_path = None

    model = get_model(model_config, checkpoint=checkpoint_path, device=device)

    if not isinstance(model, model_class):
        return None, None

    return model, model_config


def load_mindful_models(models_paths: list[Path | str],
                        model_class: Type[_MM],
                        monitor: str,
                        device: str | None = None,
                        load_weights: bool = True,
                        ) -> tuple[list[_MM], list[ModuleConfig]]:
    models: list[AbstractClassifier] = []
    model_configs: list[ModuleConfig] = []
    for model_path in models_paths:
        model, model_config = load_mindful_model(model_path, model_class, monitor, device, load_weights)
        if model is not None:
            models.append(model)
            model_configs.append(model_config)
    return models, model_configs

# endregion
