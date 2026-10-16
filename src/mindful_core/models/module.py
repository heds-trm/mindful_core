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
from mindful_core.utils.data_struct import AliasDict
from mindful_core.utils.misc import get_abstract_methods

ModelInput = torch.Tensor | list[torch.Tensor]


def unbatch_model_inputs(inputs: ModelInput) -> list[ModelInput]:
    """Unbatch a `ModelInput`, each sample becomes its own `ModelInput`.

    Parameters
    ----------
    inputs : ModelInput
        A single batched tensor (``[batch, ...]``), or a list of batched tensors
        (one per modality, each ``[batch, ...]``).

    Returns
    -------
    list[ModelInput]
        List length = Batch size. If `inputs` was a single tensor, each entry is a tensor; if a list, each
        entry is the list of per-modality tensors for that sample.
    """
    if isinstance(inputs, torch.Tensor):
        return [x for x in inputs]
    else:
        batch_size = inputs[0].shape[0]
        return [[x[i] for x in inputs] for i in range(batch_size)]


class MindfulModule(pl.LightningModule, metaclass=ABCMeta):
    """Abstract base for all Mindful modules. Extends `pl.LightningModule`.

    Provides a self-registering mechanism for subclasses. This registry is mainly intended to provide a convenient
    way to build and load models.

    `MindfulModule` provides an interface for easier optimizer/lr scheduler setup using dictionaries, an interface
    for convenient automatic logging and an interface for aggregating losses.

    Parameters
    ----------
    optimizer_config : dict[str, dict[str, Any]] | None
        A dictionary of ``{"optimizer": optimizer_params, "lr_scheduler": lr_scheduler_params}``.
        Optional if you don't intend to train the module (or if it is trained elsewhere).
        `optimizer_params` are passed to `make_optimizer` and `lr_scheduler_params` to `make_lr_scheduler`.
        `lr_scheduler_params` is optional.

    *args
        Forwarded to pl.LightningModule

    **kwargs
        Forwarded to pl.LightningModule

    Attributes
    ----------
    loss_aggregator : LossAggregator
        Accumulates losses produced this module and its submodules.
        Useful for logging and losses that are computed along the forward pass when subclassing.
    step_outputs : dict[SubsetID, list[dict[str, Any]]]
        Per-subset (`SubsetID.TRAIN`/`SubsetID.VALIDATION`/`SubsetID.TEST`) buffers that accumulates step outputs
        over an epoch.
        Cleared by at the end of each epoch.

    Notes
    -----
    The registration contract: concrete subclasses must implement `module_identifier`.
    To be registered, a class has to load or be imported somewhere.
    `get_module_class` is the lookup used elsewhere to build/load a model by identifier.

        Examples
    --------
    A minimal regression model. Subclasses provide an identifier, a ``forward``, the Lightning steps, and
    `log_epoch_outputs`. Losses go through `loss_aggregator`, and whatever each step appends to `step_outputs`
    is handed back to `log_epoch_outputs` at the end of the epoch::

        import torch
        from mindful_core.data.subset_id import SubsetID
        from mindful_core.models.module import MindfulModule

        class TinyRegressor(MindfulModule):
            def __init__(self, in_features: int, optimizer_config=None):
                super().__init__(optimizer_config=optimizer_config)
                self.linear = torch.nn.Linear(in_features, 1)

            @classmethod
            def module_identifier(cls) -> str:
                return "tiny_regressor"  # the name used to find this class later

            def forward(self, inputs):
                return self.linear(inputs)

            def base_step(self, batch, subset_id):
                self.loss_aggregator.clear()
                inputs, targets = batch
                predictions = self(inputs)

                mse_loss = torch.nn.functional.mse_loss(predictions, targets)
                self.loss_aggregator.add_loss(mse_loss, "mse")

                self.log_step_losses(subset_id)
                loss, logged_losses = self.loss_aggregator.get_logged_losses()
                self.step_outputs[subset_id].append(logged_losses)  # kept until the end of the epoch

                return loss

            def training_step(self, batch, batch_idx):
                return self.base_step(batch, SubsetID.TRAIN)

            def validation_step(self, batch, batch_idx):
                return self.base_step(batch, SubsetID.VALIDATION)

            def log_epoch_outputs(self, subset):
                outputs = self.step_outputs[subset]
                mean_mse = torch.stack([output["mse"] for output in outputs]).mean()
                self.log_epoch_metrics({"mse": mean_mse}, subset)

    Once the class is defined (or imported), it can be built from its identifier::

        model_class = MindfulModule.get_module_class("tiny_regressor")
        model = model_class(in_features=8,
                            optimizer_config={"optimizer": {"optimizer_type": "adamw", "lr": 1e-3}})
    """

    # region Class/Subclass methods
    # _module_register: dict[str, Type["MindfulModule"]] = {}
    _module_register: AliasDict[str, Type["MindfulModule"]] = AliasDict[str, Type["MindfulModule"]]()
    _abstract_named_modules: AliasDict[str, Type["MindfulModule"]] = AliasDict[str, Type["MindfulModule"]]()

    def __init_subclass__(cls, **kwargs) -> None:
        """Register classes to the module registry using their identifier and aliases.
        Concrete and abstract module classes are placed in two different registries.

        Parameters
        ----------
        **kwargs
            Forwarded to ``super().__init_subclass__`` so that any class-level keyword arguments propagate correctly
            through cooperative multiple inheritance
        """
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
        """List every identifier currently registered, across all concrete `MindfulModule` subclasses.

        Returns
        -------
        list[str]
            One entry per registered concrete module, alias mappings included.
        """
        return list(cls._module_register.keys_to_aliases_repr())

    @classmethod
    def get_registered_subclass_identifiers(cls) -> list[str]:
        """List the identifiers of registered concrete modules that are subclasses of `cls`.

        Returns
        -------
        list[str]
            Identifiers (and aliases) of registered subclasses of `cls`.
        """
        subclass_identifiers = []
        for module_identifier in cls._module_register.keys_and_aliases():
            module_class = cls._module_register[module_identifier]
            if issubclass(module_class, cls):
                subclass_identifiers.append(module_identifier)
        return subclass_identifiers

    @classmethod
    def get_module_class(cls, module_identifier: str) -> Type["MindfulModule"]:
        """Search the `MindfulModule` registry with the given `module_identifier` for a concrete module class.

        Parameters
        ----------
        module_identifier : str
            The identifier for the desired module class.

        Returns
        -------
        Type["MindfulModule"]
            The concrete `MindfulModule` class corresponding to the given `module_identifier`.

        Raises
        ------
        TypeError
            If `module_identifier` is not a string.

        KeyError
            If the given `module_identifier` does not resolve.

        NotImplementedError
            If the associated module class is abstract.

        """
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
        """Main identifier for the module class, must be unique. Required if the module class is concrete, optional
        otherwise. Will be used as a key to register this module class.

        Implementing on an abstract class can be useful to register it to the abstract named modules register, allowing
        `get_module_class` to report methods that must be implemented to turn the class concrete.

        Returns
        -------
        str
            The identifier of the module class.
        """
        raise NotImplementedError("Method `module_identifier` must be implemented in subclasses.")

    @classmethod
    def module_aliases(cls) -> tuple[str, ...]:
        """Additional identifiers for the module class. If overridden, must not collide with another module's
        identifier or aliases — at the moment this results in a silent overwrite of the previous registration(s).
        Otherwise, returns an empty tuple.

        Returns
        -------
        tuple[str, ...]
            Additional identifiers for the module class.
        """
        return ()

    @classmethod
    def multiview(cls) -> bool | int:
        """Number of augmented views this module expects per sample.

        Read by `Pipeline` to decide whether/how many augmented views to produce for each
        sample.

        Returns
        -------
        bool | int
            ``False`` for a single view (the default, no multiview). ``True`` or an `int` enables
            multiview; ``True`` resolves to 2 views.
        """
        return False

    @classmethod
    def required_sub_modules(cls) -> list[str]:
        """Names of submodules (e.g. constructor arguments) this module class requires.

        Returns
        -------
        list[str]
            Required submodule names. Empty by default.
        """
        return []

    @classmethod
    def base_log_monitors(cls) -> list[str]:
        """List base logged metric names this module uses for monitoring (i.e. in callbacks such as Lightning's
        checkpointing and early-stopping). To be extended by subclasses relying on other metrics.
        Names are before any prefix/suffix are added. Defaults to ``["loss"]``.

        Returns
        -------
        list[str]
            Metric name(s) logged by this module class for monitoring.
        """
        return ["loss"]

    @classmethod
    def get_default_log_monitors(cls,
                                 module_identifier: str,
                                 prefix: str = "validation",
                                 suffix: str = "epoch"
                                 ) -> list[str]:
        """Build the full logged metric names for a module's base monitors, following the
        ``{prefix}_{monitor}_{suffix}`` pattern. `module_identifier` is used to resolve the module to query.

        Parameters
        ----------
        module_identifier : str
            Identifier of the module class to look up, see `get_module_class`.
        prefix : str
            Prepended to each monitor name, with an underscore separator. Pass an empty
            string to disable. Defaults to ``"validation"``.
        suffix : str
            Appended to each monitor name, with an underscore separator. Pass an empty string
            to disable. Defaults to ``"epoch"``.

        Returns
        -------
        list[str]
            Full monitor names, e.g. ``"validation_loss_epoch"``.
        """
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
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
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
        """Build the optimizer (and optional LR scheduler) from ``self.optimizer_config``.

        Lightning hook, called once per training run.

        Returns
        -------
        Optimizer | tuple[list[Optimizer], list[_LRScheduler]]
            A single optimizer if ``self.optimizer_config`` has no ``"lr_scheduler"`` entry, otherwise
            the ``([optimizer], [lr_scheduler])`` pair expected by Lightning.

        Raises
        ------
        RuntimeError
            If ``self.optimizer_config`` is ``None``.
        """
        if self.optimizer_config is None:
            raise RuntimeError("When planning to train a Mindful module, you must provide an `optimizer_config`.")

        optimizer = self.make_optimizer(self.optimizer_config["optimizer"])
        if "lr_scheduler" in self.optimizer_config:
            lr_scheduler = self.make_lr_scheduler(self.optimizer_config["lr_scheduler"], optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def make_optimizer(self, config: dict[str, Any]) -> Optimizer:
        """Build an `Optimizer` from a config dict, dispatching on ``"optimizer_type"``.

        Parameters
        ----------
        config : dict[str, Any]
            Deep-copied (not mutated in place). Resolve the optimizer class with the value for ``"optimizer_type"``.
            Remaining keys are passed as kwargs to the resolved optimizer class.
            ``"optimizer_type"`` is matched case-insensitively, by substring:

            - contains ``"adamw"`` -> `AdamW`
            - contains ``"adams"`` -> `AdamS`
            - contains ``"adam"`` (otherwise) -> `Adam`, unless `config` also has a
              ``"weight_decay"`` key, which is ambiguous and raises.
            - equal to ``"nesterov"`` -> `SGD` with ``nesterov=True`` forced.

        Returns
        -------
        Optimizer
            Built over ``self.trainable_parameters``.

        Raises
        ------
        RuntimeError
            If ``"optimizer_type"`` is an ambiguous Adam variant combined with a ``"weight_decay"`` key.
        NotImplementedError
            If ``"optimizer_type"`` does not resolve.
        """
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
        """Parameters passed to the optimizer built by `make_optimizer`.

        Defaults to every parameter (``self.parameters``). Override to restrict optimization
        to a subset, e.g. when freezing a backbone.

        Returns
        -------
        Iterator[nn.Parameter]
            An iterator over the parameters passed to the optimizer built by `make_optimizer`.
        """
        return self.parameters()

    @staticmethod
    def make_lr_scheduler(config: dict[str, Any], optimizer: Optimizer) -> _LRScheduler:
        """Build a learning-rate scheduler from a config dict, dispatching on ``"scheduler_type"``.

        Parameters
        ----------
        config : dict[str, Any]
            Forwarded to `mindful_core.utils.lr_schedule.make_lr_schedule`, see that function for the full set of
            supported types.
        optimizer : Optimizer
            The optimizer the scheduler will adjust.

        Returns
        -------
        _LRScheduler
            The LR scheduler wrapping the optimizer.
        """
        return make_lr_schedule(config, optimizer)

    # endregion

    # region Logging
    def log_step_losses(self, subset_id: SubsetID) -> None:
        """Log the current total and sub-losses from ``self.loss_aggregator``.

        No-op for the test set (intended for train/validation).

        Parameters
        ----------
        subset_id : SubsetID
            Subset the current step belongs to; used as the logged metric name's prefix.
        """
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
        """Compute and log aggregate metrics for `subset` at the end of an epoch.

        Must be implemented by concrete subclasses. Called by `on_epoch_end` once per epoch and per subset. 
        Typically, reads that buffer and reports via `log_epoch_metrics`/`log_epoch_metric`.

        Parameters
        ----------
        subset : SubsetID
            Subset the epoch just completed for.

        Raises
        ------
        NotImplementedError
            If called without being overridden.
        """
        raise NotImplementedError

    def log_epoch_metrics(self,
                          metrics: dict[str, _METRIC],
                          subset: SubsetID,
                          prog_bar: bool = False,
                          logger: bool = True,
                          suffix: str = "epoch"):
        """Log the provided epoch metrics individually through `log_epoch_metric`.

        Each metric is logged as ``{subset_prefix}_{metric_name}_{suffix}``.

        Parameters
        ----------
        metrics : dict[str, _METRIC]
            Metric name to value (`torch.Tensor`/`torchmetrics.Metric`/`float`/`np.ndarray`), e.g. the output of a
            metrics summary object.
        subset : SubsetID
            Used to derive the logged name's prefix.
        prog_bar : bool
            Forwarded to ``self.log`` for every metric.
        logger : bool
            Forwarded to ``self.log`` for every metric.
        suffix : str
            Appended (with an underscore) to every metric name. Pass an empty string or ``None`` to disable.
        """
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
        """Normalize `value` to a type ``self.log`` accepts, then log it.

        Computes a `torchmetrics.Metric` if given one, reduces a multi-element tensor to its
        mean, and converts tensors/arrays to a native Python scalar via `Tensor.item`.

        Parameters
        ----------
        name : str
            Logged metric name.
        value : _METRIC
            A `float`, `torchmetrics.Metric`, `torch.Tensor`, or `np.ndarray`.
        prog_bar : bool
            Forwarded to ``self.log``.
        logger : bool
            Forwarded to ``self.log``.
        """
        if not isinstance(value, float):
            if isinstance(value, Metric):
                value = value.compute()
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    value = value.mean()
                value = value.cpu().numpy().item()
            if isinstance(value, np.ndarray):
                value = float(value)
        self.log(name=name, value=value, prog_bar=prog_bar, logger=logger)

    # endregion

    # region Hooks
    # region Register hooks
    def register_train_batch_start_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        """Register a train batch start hook on the module.

        The hook will be called every time `on_train_batch_start` is called.
        It should have the following signature::

            hook(module, batch, batch_idx) -> None

        Returns
        -------
        torch.utils.hooks.RemovableHandle
            A handle that can be used to remove the added hook by calling ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._train_batch_start_hooks)
        self._train_batch_start_hooks[handle.id] = hook
        return handle

    def register_train_batch_end_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        """Register a train batch end hook on the module.

        The hook will be called every time `on_train_batch_end` is called.
        It should have the following signature::

            hook(module, outputs, batch, batch_idx) -> None

        Returns
        -------
        torch.utils.hooks.RemovableHandle
            A handle that can be used to remove the added hook by calling ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._train_batch_end_hooks)
        self._train_batch_end_hooks[handle.id] = hook
        return handle

    def register_train_epoch_end_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        """Register a train epoch end hook on the module.

        The hook will be called every time `on_train_epoch_end` is called.
        It should have the following signature::

            hook(module) -> None

        Returns
        -------
        torch.utils.hooks.RemovableHandle
            A handle that can be used to remove the added hook by calling ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._train_epoch_end_hooks)
        self._train_epoch_end_hooks[handle.id] = hook
        return handle

    # endregion

    # region Run hooks
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        """Run registered train-batch-start hooks, then delegate to Lightning.

        See `register_train_batch_start_hook`.

        Parameters
        ----------
        batch : Any
            The input batch.
        batch_idx : int
            The batch index.

        Returns
        -------
        int | None
            Forwarded from `pl.LightningModule.on_train_batch_start`.
        """
        if self._train_batch_start_hooks:
            for hook in self._train_batch_start_hooks.values():
                hook(self, batch, batch_idx)

        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Run registered train-batch-end hooks, then delegate to Lightning.

        See `register_train_batch_end_hook`.

        Parameters
        ----------
        outputs : STEP_OUTPUT
            The output data (either a tensor or a dictionary of tensors).
        batch : Any
            The input batch.
        batch_idx : int
            The batch index.

        """
        if self._train_batch_end_hooks:
            for hook in self._train_batch_end_hooks.values():
                hook(self, outputs, batch, batch_idx)

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_before_optimizer_step(self, optimizer):
        """Log the total gradient norm, then delegate to Lightning.

        Computes the L2 norm of each trainable parameter's gradient, then the L2 norm of those norms,
        and logs it as ``"total_grad_norm"`` (epoch-level). Useful for spotting exploding/vanishing gradients.

        Parameters
        ----------
        optimizer
            Forwarded to `pl.LightningModule.on_before_optimizer_step`.
        """
        norms = {}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and (parameter.grad is not None):
                norms[name] = parameter.grad.data.detach().norm(2.0)

        total_norm = torch.tensor(list(norms.values())).norm(2.0).cpu()
        self.log("total_grad_norm", total_norm, on_step=False, on_epoch=True, logger=True)

        return super().on_before_optimizer_step(optimizer)

    def on_train_epoch_end(self) -> None:
        """Run registered train-epoch-end hooks, then ``on_epoch_end(TRAIN)``.

        See `register_train_epoch_end_hook`.
        """
        if self._train_epoch_end_hooks:
            for hook in self._train_epoch_end_hooks.values():
                hook(self)

        self.on_epoch_end(SubsetID.TRAIN)

        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        """Lightning hook; delegates to ``on_epoch_end(VALIDATION)``."""
        self.on_epoch_end(SubsetID.VALIDATION)

    def on_test_epoch_end(self) -> None:
        """Lightning hook; delegates to ``on_epoch_end(TEST)``."""
        self.on_epoch_end(SubsetID.TEST)

    def on_epoch_end(self, subset: SubsetID) -> None:
        """Report and clear ``self.step_outputs[subset]`` at the end of an epoch.

        Skipped during Lightning's sanity check. Warns if no step outputs were recorded for
        `subset` (still calls `log_epoch_outputs` regardless). ``self.step_outputs[subset]`` is
        cleared either way.

        Parameters
        ----------
        subset : SubsetID
            The subset ID to report and clear.
        """
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
        """Lightning hook; runs `register_loss_aggregators` exactly once.

        Guarded by ``self._loss_aggregator_setup``, since Lightning may call `setup` more than
        once across stages.

        Parameters
        ----------
        stage
            Forwarded by Lightning, unused.
        """
        if not self._loss_aggregator_setup:
            self.register_loss_aggregators()
            self._loss_aggregator_setup = True

    def register_loss_aggregators(self) -> None:
        """Link every nested submodule's `loss_aggregator` into ``self.loss_aggregator``.

        Walks ``self.modules`` (every nested submodule, recursively) and, for each one
        exposing a `loss_aggregator` attribute, appends it to
        ``self.loss_aggregator.sub_modules`` if not already present. This is how
        ``self.loss_aggregator`` ends up reflecting losses from submodules too.
        """
        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "loss_aggregator"):
                if module.loss_aggregator not in self.loss_aggregator:
                    self.loss_aggregator.sub_modules.append(module.loss_aggregator)

    # endregion
