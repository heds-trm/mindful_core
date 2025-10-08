from pathlib import Path
from enum import Enum
from typing import Any, TypedDict

from mindful_core.utils.parsing import parse_list, parse_checkpoint_path, parse_batch_size
from mindful_core.utils.misc import try_load_json, write_json
from mindful_core.data.subset_id import SubsetID
from mindful_core.data.data_folds import DataFold, PresetFold
from mindful_core.models.index import ModuleConfig, MindfulModule
from mindful_core.experiments.experiment_stage import ExperimentStages


class GradientClipConfig(TypedDict):
    value: float | None
    algorithm: str | None


class ExperimentRoundConfig(object):
    def __init__(self,
                 stages: str | list[str] | dict[str, Any],
                 log_dir: str,
                 model: ModuleConfig,
                 pipeline_config: str,

                 checkpoint: str | Path | None = None,
                 gradient_clip_config: GradientClipConfig | None = None,
                 preparation_pipeline_export: str | None = None,

                 resume_training: bool = False,
                 use_last: bool = False,
                 save_last: bool | None = None,

                 fold_path: str | None = None,
                 ref_fold_path: str | None = None,
                 scalar_features_path: str | None = None,
                 categorical_features_path: str | None = None,

                 model_dir: str | None = None,
                 checkpoint_dir: str | None = None,
                 training_data_reduction: float | None = None,

                 batch_size: int | None = None,
                 train_batch_size: int | str | None = None,
                 test_batch_size: int | str | None = None,
                 use_imbalanced_sampler: bool | None = None,
                 max_epochs: str | None = None,

                 validate_every_n_epochs: int = 1,
                 log_monitors: list[str] | None = None,
                 load_monitor: str | None = None,
                 milestones: str | None = None,
                 patience: str | None = None,

                 visualizers: dict[str, dict[str, Any]] | None = None,
                 ww_frequency: int | None = None,

                 num_workers: str | None = None,
                 accelerator: str | None = None,
                 devices: int | None = None,
                 seed: str | int | None = None,
                 ):
        self._capture_argument_values = False
        self._argument_values = {}
        self._capture_argument_values = True

        self.stages = ExperimentStages(stages)

        # region Models / Checkpoints
        self.model = model
        self.load_monitor = load_monitor
        self.checkpoint = parse_checkpoint_path(checkpoint, monitor=load_monitor, use_last=use_last, verbose=True)
        self.resume_training = resume_training

        # self.rep_model = rep_model.lower() if rep_model is not None else None
        # self.rep_checkpoint = parse_checkpoint_path(rep_checkpoint, monitor=load_monitor, use_last=use_last,
        #                                             verbose=True)
        self.use_last = use_last
        self.save_last = save_last
        # endregion

        # region Input data
        # ## Note: In Sagemaker, data is stored in /opt/ml/input/data/[channel_name]
        self.fold_path = fold_path or self.get_default_reference_fold()
        self.ref_fold_path = ref_fold_path
        self.scalar_features_path = scalar_features_path
        self.categorical_features_path = categorical_features_path
        self.training_data_reduction = training_data_reduction
        self.use_imbalanced_sampler = use_imbalanced_sampler
        # endregion

        self.preparation_pipeline_export = preparation_pipeline_export
        self.pipeline_config_path = pipeline_config
        self.base_pipeline_config = try_load_json(pipeline_config, "Base Pipeline config")

        # region Directories
        self.log_dir: str = log_dir
        # ## Note: In Sagemaker, configs and partitions should be put with the code as they will be uploaded with it
        self.model_dir: str | None = model_dir
        self.checkpoint_dir: str | None = checkpoint_dir
        # endregion

        self.gradient_clip_config = gradient_clip_config

        # region Batch sizes
        train_batch_size = parse_batch_size(train_batch_size, batch_size)
        test_batch_size = parse_batch_size(test_batch_size, batch_size)

        self.batch_sizes = {
            SubsetID.TRAIN: train_batch_size,
            SubsetID.VALIDATION: test_batch_size,
            SubsetID.TEST: test_batch_size,
        }
        # endregion

        # region Milestones / Max epochs
        self.milestones = None if milestones is None else parse_list(milestones)
        self.max_epochs = None if max_epochs is None else int(max_epochs)
        self.validate_every_n_epochs = validate_every_n_epochs

        if log_monitors is None:
            log_monitors = MindfulModule.get_default_log_monitors()

        self.log_monitors = log_monitors
        self.patience = None if patience is None else int(patience)

        if (self.milestones is not None) and (len(self.milestones) == 0):
            self.milestones = None

        if self.max_epochs is None:
            if self.milestones is None:
                if self.stages.include_training:
                    raise ValueError("During training, at least one of `milestones`, `max_epochs` "
                                     "and `epochs` must be provided.")
            else:
                self.max_epochs = max(self.milestones)
        else:
            if self.milestones is not None:
                self.max_epochs = max(self.max_epochs, *self.milestones)
        # endregion

        self.visualizers_config = visualizers
        self.ww_frequency = ww_frequency
        self.num_workers = int(num_workers) if num_workers is not None else None
        self.accelerator: str = accelerator or "cpu"
        self.devices = devices
        self.seed = int(seed) if seed is not None else None

        self._capture_argument_values = False
        self._fold: DataFold | None = None

    def __repr__(self):
        return self._argument_values.__repr__()

    # region Properties
    @property
    def model_class_name(self) -> str:
        return self.model["class_name"]

    @property
    def model_hparams(self) -> dict[str, Any]:
        return self.model["hparams"]

    @property
    def default_model_hparams_path(self) -> Path:
        return self.get_default_hparams_filepath(self.model_class_name)

    def get_default_hparams_filepath(self, model_name: str) -> Path:
        folder = self.get_default_reference_folder()
        if folder is None:
            raise RuntimeError("Could not find a default reference folder to search in "
                               "for finding a default hparams file.")
        return Path(folder, "{}_hparams.json".format(model_name))

    def get_default_reference_fold(self) -> Path | None:
        folder = self.get_default_reference_folder()
        if folder is None:
            return None

        reduced_fold_path = Path(folder, "reduced_fold.csv")
        if reduced_fold_path.exists():
            return reduced_fold_path
        else:
            return Path(folder, "fold.csv")

    def get_default_reference_folder(self) -> Path | None:
        if self.checkpoint is not None:
            folder = self.checkpoint
        else:
            return None

        return Path(folder).parent.parent

    @property
    def gradient_clip_value(self) -> float | None:
        if self.gradient_clip_config is None:
            return None

        return self.gradient_clip_config.get("value")

    @property
    def gradient_clip_algorithm(self) -> float | None:
        if self.gradient_clip_config is None:
            return None

        return self.gradient_clip_config.get("algorithm")

    # endregion

    # region Folds
    @property
    def fold(self) -> PresetFold:
        if self._fold is None:
            print("PresetFold - Loading fold from: {}".format(self.fold_path))
            self._fold = PresetFold(self.fold_path,
                                    seed=self.seed,
                                    scalar_features_path=self.scalar_features_path,
                                    categorical_features_path=self.categorical_features_path)
        return self._fold

    # endregion

    # region Save / Load (hparams, properties, ...)

    def __setattr__(self, key, value):
        if key != "_capture_argument_values":
            if self._capture_argument_values:
                self._argument_values[key] = to_serializable(value)
        super(ExperimentRoundConfig, self).__setattr__(key, value)

    def save(self, directory: str):
        # region Experiment config
        experiment_config_path = Path(directory, "experiment_config.json")
        write_json(experiment_config_path, self._argument_values)
        # endregion

        # region Model hparams
        model_config_path = Path(directory, "{}_config.json".format(self.model_class_name))
        write_json(model_config_path, self.model)

        model_hparams_path = Path(directory, "{}_hparams.json".format(self.model_class_name))
        write_json(model_hparams_path, self.model_hparams)
        # endregion

        # region Fold
        fold_path = Path(directory, "fold.csv")
        self.fold.save_scan_data(fold_path)
        # endregion

        # region Environment packages
        import importlib_metadata
        environment_path = Path(directory, "environment.txt")
        with open(environment_path, 'w') as output_environment_file:
            distributions = importlib_metadata.distributions()
            environment = ["{}=={}\n".format(distribution.name, distribution.version)
                           for distribution in distributions]
            output_environment_file.writelines(environment)
        # endregion

    # endregion


def to_serializable(value: Any) -> Any:
    # iterables
    if isinstance(value, tuple):
        return tuple([to_serializable(x) for x in value])
    elif isinstance(value, list):
        return [to_serializable(x) for x in value]
    elif isinstance(value, dict):
        return {to_serializable(x): to_serializable(y) for x, y in value.items()}
    # elements
    elif isinstance(value, Enum):
        return str(value)
    elif isinstance(value, Path):
        return value.as_posix()
    elif isinstance(value, ExperimentStages):
        return value.serialize()
    else:
        return value
