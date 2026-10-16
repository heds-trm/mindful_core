import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from typing import Sequence


class EpochProgressBar(TQDMProgressBar):
    def __init__(self, process_position: int = 0):
        super(EpochProgressBar, self).__init__(refresh_rate=0, process_position=process_position)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _should_update(self, current: int, total: int) -> bool:
        return False


class MilestoneCheckpoint(ModelCheckpoint):
    def __init__(self,
                 milestones: Sequence[int],
                 dirpath: str | None = None,
                 filename: str | None = None,
                 monitor: str | None = None,
                 verbose: bool = False,
                 save_last: bool | None = None,
                 save_top_k: int = 1,
                 save_weights_only: bool = False,
                 mode: str = "min",
                 auto_insert_metric_name: bool = True,
                 every_n_train_steps: int | None = None,
                 train_time_interval: int | None = None,
                 every_n_epochs: int | None = None,
                 save_on_train_epoch_end: bool | None = None,
                 ):
        super(MilestoneCheckpoint, self).__init__(dirpath=dirpath, filename=filename,
                                                  monitor=monitor, verbose=verbose,
                                                  save_last=save_last, save_top_k=save_top_k,
                                                  save_weights_only=save_weights_only, mode=mode,
                                                  auto_insert_metric_name=auto_insert_metric_name,
                                                  every_n_train_steps=every_n_train_steps,
                                                  train_time_interval=train_time_interval,
                                                  every_n_epochs=every_n_epochs,
                                                  save_on_train_epoch_end=save_on_train_epoch_end)
        self.milestones = milestones

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save a checkpoint at the end of the training epoch."""
        super(MilestoneCheckpoint, self).on_train_epoch_end(trainer, pl_module)
        if self._save_on_train_epoch_end:
            self._save_checkpoint_at_milestone(trainer)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save a checkpoint at the end of the validation stage."""
        super(MilestoneCheckpoint, self).on_validation_end(trainer, pl_module)
        if not self._save_on_train_epoch_end:
            self._save_checkpoint_at_milestone(trainer)

    def _save_checkpoint_at_milestone(self, trainer: pl.Trainer):
        current_epoch = trainer.current_epoch + 1
        if current_epoch in self.milestones:
            filepath = os.path.join(self.dirpath, "model_milestone_{}.ckpt".format(current_epoch))
            self._save_checkpoint(trainer, filepath)
            print("Saved milestone n{} at {}.".format(current_epoch, filepath))
