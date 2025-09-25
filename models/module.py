import torch
import torch.nn as nn
import torch.utils.hooks as hooks
import pytorch_lightning as pl
import numpy as np
# noinspection PyUnresolvedReferences, PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer, Adam, AdamW, SGD
# noinspection PyUnresolvedReferences, PyProtectedMember
from pytorch_lightning.utilities.types import _METRIC, STEP_OUTPUT
from torchmetrics.metric import Metric
import warnings
import copy
from collections import OrderedDict
from abc import abstractmethod, ABCMeta
import inspect
from typing import Any, Iterator, Callable, Type

from mindful_core.data.subset_id import SubsetID
from mindful_core.models.loss_aggregator import LossAggregator
from mindful_core.utils.lr_schedule import make_lr_schedule
from mindful_core.utils.adams import AdamS
from mindful_core.utils.struct import AliasDict
from mindful_core.utils.misc import get_abstract_methods

ModelInput = torch.Tensor | list[torch.Tensor]


def unbatch_model_inputs(inputs: ModelInput) -> list[ModelInput]:
    if isinstance(inputs, torch.Tensor):
        return [x for x in inputs]
    else:
        batch_size = inputs[0].shape[0]
        return [[x[i] for x in inputs] for i in range(batch_size)]


class MindfulModule(pl.LightningModule, metaclass=ABCMeta):
    """
        todo: write documentation
    """
    # region Class/Subclass methods
    # _module_register: dict[str, Type["MindfulModule"]] = {}
    _module_register: AliasDict[str, Type["MindfulModule"]] = AliasDict()
    _abstract_named_modules: AliasDict[str, Type["MindfulModule"]] = AliasDict()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        try:
            identifier = cls.module_identifier()
            register = cls._module_register if not inspect.isabstract(cls) else cls._abstract_named_modules
            register[identifier] = cls
            aliases = cls.module_aliases()
            if len(aliases) > 0:
                register.add_aliases(identifier, *aliases)

        except NotImplementedError:
            pass

    @classmethod
    def known_module_identifiers(cls) -> list[str]:
        return list(cls._module_register.keys_to_aliases_repr())

    @classmethod
    def get_registered_subclass_identifiers(cls) -> list[str]:
        subclass_identifiers = []
        for module_identifier in cls._module_register.keys_and_aliases():
            module_class = cls._module_register[module_identifier]
            if issubclass(module_class, cls):
                subclass_identifiers.append(module_identifier)
        return subclass_identifiers

    @classmethod
    def get_module_class(cls, module_identifier: str) -> Type["MindfulModule"]:
        if not isinstance(module_identifier, str):
            raise TypeError("Expected `module_identifier` to be a string, got a {} ({})".
                            format(type(module_identifier), module_identifier))

        if module_identifier not in cls._module_register:
            if module_identifier in cls._abstract_named_modules:
                match = cls._abstract_named_modules[module_identifier]
                abstract_methods = get_abstract_methods(match)
                raise NotImplementedError("Module identifier `{}` was found (for class `{}`) but the corresponding "
                                          "class is abstract. Make sure all abstract methods are implemented first. "
                                          "The following methods are abstract: `{}`.".
                                          format(module_identifier, match, abstract_methods))
            else:
                raise KeyError("Unknown module identifier `{}`. Modules found are: {}. "
                               "If your module is not in the list, make sure it was imported in python first.".
                               format(module_identifier, cls.known_module_identifiers()))
        return cls._module_register[module_identifier]

    @classmethod
    @abstractmethod
    def module_identifier(cls) -> str:
        raise NotImplementedError("Method `module_identifier` must be implemented in subclasses.")

    @classmethod
    def module_aliases(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def multiview(cls) -> bool | int:
        return False

    @classmethod
    def required_sub_modules(cls) -> list[str]:
        return []

    @classmethod
    def base_log_monitors(cls) -> list[str]:
        return ["loss"]

    @classmethod
    def get_default_log_monitors(cls,
                                 module_identifier: str,
                                 prefix: str = "validation",
                                 suffix: str = "epoch"
                                 ) -> list[str]:
        module_class = cls.get_module_class(module_identifier)

        if len(prefix) > 0:
            prefix = prefix + "_"

        if len(suffix) > 0:
            suffix = "_" + suffix

        log_monitors = [prefix + log_monitor + suffix for log_monitor in module_class.base_log_monitors()]
        return log_monitors

    # endregion

    def __init__(self,
                 optimizer_config: dict[str, dict[str, Any]] | None = None,
                 *args, **kwargs):
        super(MindfulModule, self).__init__(*args, **kwargs)
        self.optimizer_config = optimizer_config
        self.loss_aggregator = LossAggregator()
        self._loss_aggregator_setup = False

        self.step_outputs: dict[SubsetID, list[dict[str, Any]]] = {
            SubsetID.TRAIN: [],
            SubsetID.VALIDATION: [],
            SubsetID.TEST: [],
        }

        self._train_batch_start_hooks: dict[int, Callable] = OrderedDict()
        self._train_batch_end_hooks: dict[int, Callable] = OrderedDict()
        self._train_epoch_end_hooks: dict[int, Callable] = OrderedDict()

        self._args = args
        self._kwargs = kwargs

    # region Optimizers / Schedulers
    def configure_optimizers(self):
        if self.optimizer_config is None:
            raise RuntimeError("When planning to train a Mindful module, you must provide an `optimizer_config`.")

        optimizer = self.make_optimizer(self.optimizer_config["optimizer"])
        if "lr_scheduler" in self.optimizer_config:
            lr_scheduler = self.make_lr_scheduler(self.optimizer_config["lr_scheduler"], optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def make_optimizer(self, config: dict[str, Any]) -> Optimizer:
        config = copy.deepcopy(config)
        optimizer_type: str = config.pop("optimizer_type").lower()
        if "adam" in optimizer_type:
            if "adamw" in optimizer_type:
                optimizer_class = AdamW

            elif "adams" in optimizer_type:
                optimizer_class = AdamS

            elif "weight_decay" in config:
                raise RuntimeError("Could not determine which Adam with "
                                   "weight decay to use from {}.".format(optimizer_type))
            else:
                optimizer_class = Adam

        elif optimizer_type == "nesterov":
            optimizer_class = SGD
            config["nesterov"] = True

        else:
            raise NotImplementedError("Optimizer `{}` is either unknown or has not been implemented yet."
                                      .format(optimizer_type))

        return optimizer_class(params=self.trainable_parameters, **config)

    @property
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    @staticmethod
    def make_lr_scheduler(config: dict[str, Any], optimizer: Optimizer) -> _LRScheduler:
        return make_lr_schedule(config, optimizer)

    # endregion

    # region Logging
    def log_step_losses(self, subset_id: SubsetID) -> None:
        if subset_id == SubsetID.TEST:
            return

        # on_step = subset_id == SubsetID.TRAIN
        on_step = False
        prefix = subset_id.as_prefix()

        total_loss = self.loss_aggregator.get_total_loss()
        self.log("{}_loss".format(prefix), total_loss, on_step=on_step, on_epoch=True, logger=True)

        for loss_key, loss_value in self.loss_aggregator.get_losses().items():
            log_name = "{}_{}".format(prefix, loss_key)
            self.log(log_name, loss_value, on_step=on_step, on_epoch=True, logger=True)

    @abstractmethod
    def log_epoch_outputs(self, subset: SubsetID):
        raise NotImplementedError

    def log_epoch_metrics(self,
                          metrics: dict[str, _METRIC],
                          subset: SubsetID,
                          prog_bar: bool = False,
                          logger: bool = True,
                          suffix: str = "epoch"):
        prefix = subset.as_prefix()
        suffix = "" if suffix is None else "_{}".format(suffix)
        for metric_name in metrics:
            log_entry_name = "{}_{}{}".format(prefix, metric_name, suffix)
            self.log_epoch_metric(log_entry_name, metrics[metric_name], logger=logger, prog_bar=prog_bar)

    def log_epoch_metric(self,
                         name: str,
                         value: _METRIC,
                         prog_bar: bool = False,
                         logger: bool = True,
                         ) -> None:
        if not isinstance(value, float):
            if isinstance(value, Metric):
                value = value.compute()
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    value = value.mean()
                value = value.cpu().numpy()
            if isinstance(value, np.ndarray):
                value = float(value)
        self.log(name=name, value=value, prog_bar=prog_bar, logger=logger)

    # endregion

    # region Hooks
    # region Register hooks
    def register_train_batch_start_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        r"""Registers a train batch start hook on the module.

        The hook will be called every time :func:`on_train_batch_start` is called.
        It should have the following signature::

            hook(module, batch, batch_idx) -> None

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._train_batch_start_hooks)
        self._train_batch_start_hooks[handle.id] = hook
        return handle

    def register_train_batch_end_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        r"""Registers a train batch end hook on the module.

        The hook will be called every time :func:`on_train_batch_end` is called.
        It should have the following signature::

            hook(module, outputs, batch, batch_idx) -> None

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._train_batch_end_hooks)
        self._train_batch_end_hooks[handle.id] = hook
        return handle

    def register_train_epoch_end_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        r"""Registers a train epoch end hook on the module.

        The hook will be called every time :func:`on_train_epoch_end` is called.
        It should have the following signature::

            hook(module) -> None

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._train_epoch_end_hooks)
        self._train_epoch_end_hooks[handle.id] = hook
        return handle

    # endregion

    # region Run hooks
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        if self._train_batch_start_hooks:
            for hook in self._train_batch_start_hooks.values():
                hook(self, batch, batch_idx)

        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self._train_batch_end_hooks:
            for hook in self._train_batch_end_hooks.values():
                hook(self, outputs, batch, batch_idx)

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_before_optimizer_step(self, optimizer):
        norms = {
            name: parameter.grad.data.detach().norm(2.0)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and (parameter.grad is not None)
        }

        total_norm = torch.tensor(list(norms.values())).norm(2.0).cpu()
        self.log("total_grad_norm", total_norm, on_step=False, on_epoch=True, logger=True)

        return super().on_before_optimizer_step(optimizer)

    def on_train_epoch_end(self) -> None:
        if self._train_epoch_end_hooks:
            for hook in self._train_epoch_end_hooks.values():
                hook(self)

        self.on_epoch_end(SubsetID.TRAIN)

        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end(SubsetID.VALIDATION)

    def on_test_epoch_end(self) -> None:
        self.on_epoch_end(SubsetID.TEST)

    def on_epoch_end(self, subset: SubsetID) -> None:
        if not self.trainer.sanity_checking:
            if len(self.step_outputs[subset]) == 0:
                warnings.warn("No data found in `self.step_outputs` for `{}` "
                              "subset in module of type `{}`".format(subset, type(self)))
            self.log_epoch_outputs(subset)
        self.step_outputs[subset].clear()

    # endregion
    # endregion

    # region Dynamic setup
    def setup(self, stage):
        if not self._loss_aggregator_setup:
            self.register_loss_aggregators()
            self._loss_aggregator_setup = True

    def register_loss_aggregators(self) -> None:
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "loss_aggregator"):
                if module.loss_aggregator not in self.loss_aggregator:
                    self.loss_aggregator.sub_modules.append(module.loss_aggregator)

    # endregion
