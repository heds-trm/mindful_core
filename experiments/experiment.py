from monai.utils import set_determinism
from pytorch_lightning import seed_everything
import pandas as pd
import shutil
from pathlib import Path
import copy
from typing import Any

from mindful_core.utils.data_constants import SCAN_ID
from mindful_core.utils.misc import try_load_json
from mindful_core.experiments.experiment_round_config import ExperimentRoundConfig, GradientClipConfig
from mindful_core.experiments.experiment_round import ExperimentRound


class Experiment(object):
    def __init__(self,
                 base_config: dict,
                 folds: dict[int, str],
                 seeds: list[int] | None,
                 checkpoint: str | Path | None = None,
                 name: str | None = None,
                 ) -> None:
        self.base_config = base_config
        self.folds = folds
        self.seeds = seeds
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.name = name

        if (seeds is not None) and (len(seeds) < len(folds)):
            raise ValueError("You have more folds than seeds ({} > {})".format(len(folds), len(seeds)))

        self.gradient_clip_config: GradientClipConfig | None = None

        self._load_model_hparams()
        self._pop_and_set_gradient_clipping()

    def run(self):
        # TODO: Save config in experiment folder
        for fold_index, fold_path in self.folds.items():
            self.print_current_step_and_fold(fold_index)
            round_config = copy.deepcopy(self.base_config)

            seed = self.get_seed_for_round(fold_index)
            if seed is not None:
                set_determinism(seed)
                seed_everything(seed)
                round_config["seed"] = seed

            # TODO : Allow for a dictionary of checkpoints to allow for importing models from other experiments
            #   as sub-modules (eg. import an encoder trained with SSL in an encoder+mlp setup)
            round_config["fold_path"] = fold_path
            round_config["checkpoint"] = self.get_checkpoint_for_round(fold_index)
            round_config["gradient_clip_config"] = self.gradient_clip_config

            experiment_round_config = ExperimentRoundConfig(**round_config)
            experiment_round = ExperimentRound(experiment_round_config)
            experiment_round.run()

        if self.has_test_stage:
            self.gather_inferences()

    def _load_model_hparams(self) -> None:
        hparams = self.hparams

        if "additional_model_hparams" in self.base_config:
            additional_model_hparams: dict[str, Any] = self.base_config.pop("additional_model_hparams")
            hparams = merge_hparams(hparams, additional_model_hparams)

        self.hparams = hparams

    def _pop_and_set_gradient_clipping(self):
        if "optimizer_config" not in self.hparams:
            return

        optimizer_config: dict[str, Any] = self.hparams["optimizer_config"]

        if "gradient_clip_val" in optimizer_config:
            gradient_clip_val = optimizer_config.pop("gradient_clip_val")
        else:
            gradient_clip_val = None

        if "gradient_clip_algorithm" in optimizer_config:
            gradient_clip_algorithm = optimizer_config.pop("gradient_clip_algorithm")
        else:
            gradient_clip_algorithm = None

        self.hparams["optimizer_config"] = optimizer_config
        self.gradient_clip_config = {
            "value": gradient_clip_val,
            "algorithm": gradient_clip_algorithm
        }

    def get_seed_for_round(self, fold_index: int) -> int | None:
        if self.seeds is None:
            return None
        
        if len(self.seeds) < len(self.folds):
            raise RuntimeError("You have more folds than seeds ({} > {}). "
                               "This should have been verified during initialisation, did you change seeds or folds ?".
                               format(len(self.folds), len(self.seeds)))

        if fold_index >= len(self.seeds):
            raise ValueError("Fold index `{}` is out of bounds for seeds ({} seeds available).".
                             format(fold_index, len(self.seeds)))

        return self.seeds[fold_index]

    def get_checkpoint_for_round(self, fold_index: int) -> Path | None:
        if self.checkpoint is None:
            return None

        if self.checkpoint.exists() and self.checkpoint.is_file():
            return self.checkpoint

        checkpoint = self.checkpoint / "version_{}".format(fold_index)
        if not checkpoint.exists():
            checkpoint = self.checkpoint / "version_0"

            if not checkpoint.exists():
                raise FileNotFoundError("Could not find any valid checkpoint version in `{}`.".format(self.checkpoint))

        return checkpoint

    def gather_inferences(self) -> None:
        lightning_logs_dir = self.log_dir / "lightning_logs"

        test_inferences: list[pd.DataFrame] | pd.DataFrame = []
        for filepath in lightning_logs_dir.iterdir():
            if filepath.match("version_*") and filepath.is_dir():
                fold_inferences_path = filepath / "inferences.csv"
                fold_id = int(filepath.stem.split("_")[-1])
                target_inferences_path = self.log_dir / "inferences_fold_{:02d}.csv".format(fold_id)
                if fold_inferences_path.exists():
                    shutil.copy2(fold_inferences_path, target_inferences_path)

                fold_test_inferences_path = filepath / "test_inferences.csv"
                if fold_test_inferences_path.exists():
                    fold_test_inferences = pd.read_csv(fold_test_inferences_path, index_col=SCAN_ID)
                    test_inferences.append(fold_test_inferences)

        if len(test_inferences) == 0:
            return

        test_inferences: pd.DataFrame = pd.concat(test_inferences)
        output_filepath = self.log_dir / "test_inferences.csv"
        test_inferences.to_csv(output_filepath.as_posix(), index_label=SCAN_ID)

    def print_current_step_and_fold(self, fold_index: int) -> None:
        if self.name is None:
            # Verbose is disabled when no name is provided
            return

        text = "|| Starting round for `{}`: Fold `{}`||".format(self.name, fold_index)
        print("=" * len(text))
        print(text)
        print("=" * len(text))

    @property
    def hparams(self) -> dict[str, Any]:
        hparams = self.base_config["model"]["hparams"]
        if isinstance(hparams, (str, Path)):
            hparams = try_load_json(hparams, "Hparams")
            self.hparams = hparams

        return hparams

    @hparams.setter
    def hparams(self, value: dict[str, Any]) -> None:
        self.base_config["model"]["hparams"] = value

    @property
    def log_dir(self) -> Path:
        return self.base_config["log_dir"]

    @property
    def has_test_stage(self) -> bool:
        return "test" in self.base_config["stages"]


def merge_hparams(hparams: dict[str, Any] | str | Path, additional_hparams: dict[str, Any]) -> dict[str, Any]:
    if isinstance(hparams, (str, Path)):
        hparams = try_load_json(hparams, "Base hparams")

    result = copy.deepcopy(hparams)
    for key in additional_hparams:
        if key in result:
            if isinstance(result[key], dict) and isinstance(additional_hparams[key], dict):
                # Merge dictionaries
                result[key] = merge_hparams(result[key], additional_hparams[key])
            else:
                # Override base hparams with additional hparams
                result[key] = additional_hparams[key]
        else:
            # Insert new values
            result[key] = additional_hparams[key]

    return result
