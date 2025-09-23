import torch.nn as nn
from torch.optim import Optimizer, Adam
# noinspection PyUnresolvedReferences, PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts, OneCycleLR
import math
import warnings
from typing import Any

from utils.tensor_utils import lerp

"""
Implementation taken from pytorch-lightning-bolts, extracted because pl_bolts is not up-to-date with pytorch-lightning.
https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/optimizers/lr_scheduler.py
"""


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.

    . warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    . warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 0.0,
            eta_min: float = 0.0,
            last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                    1 +
                    math.cos(
                        math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> list[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


class CosineAnnealingDecayLR(_LRScheduler):
    base_lrs: list[float]

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: int,
                 max_epochs: int,
                 decay=0.99,
                 warmup_start_lr: float = 0.0,
                 eta_min: float = 0.0,
                 last_epoch: int = -1):
        self.last_epoch = last_epoch
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.decay = decay
        super(CosineAnnealingDecayLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch == 0:
            learning_rates = [self.warmup_start_lr] * len(self.base_lrs)

        elif self.last_epoch < self.warmup_epochs:
            delta_lrs = [(base_lr - self.warmup_start_lr) for base_lr in self.base_lrs]
            step_lrs = [delta_lr / (self.warmup_epochs - 1) for delta_lr in delta_lrs]
            learning_rates = [step_lr * self.last_epoch for step_lr in step_lrs]

        else:
            cosine_last_epoch = self.last_epoch - self.warmup_epochs
            cosine_max_epoch = self.max_epochs - self.warmup_epochs
            progress = cosine_last_epoch / cosine_max_epoch
            factor = (math.cos(math.pi * progress) + 1.0) / 2.0
            decay = self.decay ** cosine_last_epoch
            learning_rates = [lerp(self.eta_min, base_lr, factor) * decay for base_lr in self.base_lrs]

        return learning_rates

    def _get_closed_form_lr(self) -> list[float]:
        return self.get_lr()


class CosineDecayLR(CosineAnnealingDecayLR):
    def get_lr(self) -> list[float]:
        if self.last_epoch > self.max_epochs:
            return [self.eta_min] * len(self.base_lrs)
        else:
            return super(CosineDecayLR, self).get_lr()


class CosineAnnealingDecayRestarts(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 initial_warmup_epochs: int,
                 initial_restart_epochs: int,
                 restart_decay=0.5,
                 restart_max_epochs_multiplier=2,
                 warmup_start_lr: float = 0.0,
                 eta_min: float = 0.0,
                 last_epoch: int = -1):
        self.last_epoch = last_epoch
        self.eta_min = eta_min
        self.initial_restart_epochs = initial_restart_epochs
        self.initial_warmup_epochs = initial_warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.restart_decay = restart_decay
        self.restart_max_epochs_multiplier = restart_max_epochs_multiplier
        super(CosineAnnealingDecayRestarts, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch == 0:
            learning_rates = [self.warmup_start_lr] * len(self.base_lrs)

        elif self.last_epoch < self.initial_warmup_epochs:
            delta_lrs = [(base_lr - self.warmup_start_lr) for base_lr in self.base_lrs]
            step_lrs = [delta_lr / (self.initial_warmup_epochs - 1) for delta_lr in delta_lrs]
            learning_rates = [step_lr * self.last_epoch for step_lr in step_lrs]

        else:
            progress, decay = self.get_current_restart_factors()
            factor = (math.cos(math.pi * progress) + 1.0) / 2.0
            learning_rates = [lerp(self.eta_min, base_lr, factor) * decay for base_lr in self.base_lrs]

        return learning_rates

    def _get_closed_form_lr(self) -> list[float]:
        return self.get_lr()

    def get_current_restart_factors(self) -> tuple[float, float]:
        """
        :return: A tuple of 1) the progress of the current restart and 2) the decay associated with this restart.
        """
        if self.last_epoch < self.initial_warmup_epochs:
            raise RuntimeError

        restart_last_epoch = self.last_epoch - self.initial_warmup_epochs

        if self.restart_max_epochs_multiplier <= 1:
            restart_index = restart_last_epoch // self.initial_restart_epochs
            progress = restart_last_epoch / self.initial_restart_epochs - restart_index
            decay = self.restart_decay ** restart_index
            return progress, decay

        restart_max_epochs = self.initial_restart_epochs
        restart_index = 0
        while restart_last_epoch > restart_max_epochs:
            restart_index += 1
            restart_last_epoch -= restart_max_epochs
            restart_max_epochs *= self.restart_max_epochs_multiplier

        progress = restart_last_epoch / restart_max_epochs
        decay = self.restart_decay ** restart_index
        return progress, decay


def make_lr_schedule(config: dict[str, Any], optimizer: Optimizer) -> _LRScheduler:
    schedule_type: str = config.pop("scheduler_type")
    schedule_type = schedule_type.lower()
    if "cosine" in schedule_type:
        if "decay" in schedule_type:
            if "restart" in schedule_type:
                return CosineAnnealingDecayRestarts(optimizer=optimizer, **config)
            else:
                if "annealing" in schedule_type:
                    return CosineAnnealingDecayLR(optimizer=optimizer, **config)
                else:
                    return CosineDecayLR(optimizer=optimizer, **config)
        else:
            if "restart" in schedule_type:
                # noinspection PyTypeChecker
                return CosineAnnealingWarmRestarts(optimizer=optimizer, **config)
            else:
                return LinearWarmupCosineAnnealingLR(optimizer=optimizer, **config)
    elif schedule_type == "one_cycle":
        # noinspection PyTypeChecker
        return OneCycleLR(optimizer=optimizer, **config)
    else:
        raise ValueError("LR Schedule `{}` is either unknown or has not been implemented yet.".format(schedule_type))
