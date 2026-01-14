from pathlib import Path
import copy
import warnings
from typing import TypedDict, Literal, Any

from mindful_core.utils.misc import try_load_json
from mindful_core.models.index import ModuleConfig
from mindful_core.data.datasets.protoset import AbstractProtoSet, get_proto_datasets
from mindful_core.experiments.experiment import Experiment
from mindful_core.scripts.outcomes.gather_test_results import gather_test_results


# TODO: Improve dataset configs so that merged dataset have their own name and can reference and be built upon single
#   dataset. Partitions should be either taken as-is by default, or re-attributed to train/val/test according to a dict
class DatasetConfig(TypedDict, total=False):
    folds: str

    scalar_features: str
    categorical_features: str

    preparation_pipeline: str
    prepared_folds: str
    preparation_pipeline_export: str


class ExperimentConfig(TypedDict, total=False):
    model: ModuleConfig
    checkpoint: str

    log_monitors: list[str]

    batch_size: int | str
    pipeline_config: str

    max_epochs: int
    patience: int

    stages: str | list[str] | dict[str, Any]
    seeds: list[int]

    visualizers: str

    num_workers: int
    accelerator: str
    devices: int

    skip: Literal["no", "yes"]
    dataset: str
    folds: str | int | list[int]
    based_on: str


class ExperimentSeriesConfig(TypedDict, total=False):
    datasets: dict[str, DatasetConfig]
    experiments: dict[str, ExperimentConfig]
    shared: dict[str, ExperimentConfig]
    log_dir: str


class ExperimentSeries(object):
    def __init__(self, config: str | Path | dict | ExperimentSeriesConfig) -> None:
        if isinstance(config, (str, Path)):
            config = try_load_json(config, "Experiment Series config")

        config = copy.deepcopy(config)
        self._check_config(config)

        config["shared"] = self.resolve_shared_configs(config["shared"])

        self.config: ExperimentSeriesConfig = config
        self.log_dir = Path(config["log_dir"])

    @staticmethod
    def _check_config(config: ExperimentSeriesConfig) -> None:
        if "log_dir" not in config:
            raise RuntimeError("`log_dir` is missing from experiment series configuration.")

        # TODO : Check every verifiable aspect of every configuration, aggregate all errors and display them
        warnings.warn("ExperimentSeries._check_config: Currently not fully implemented, "
                      "configurations are not being checked before they are used.")

    @staticmethod
    def resolve_shared_configs(shared: dict[str, ExperimentConfig]) -> dict[str, ExperimentConfig]:
        """
            If some shared configs are based on other shared configs,
                copies their parameters and removes the reference to their base.
        Args:
            shared: A dictionary of experiment configs.

        Returns:
            A dictionary of experiment configs without references to other configs.

        """
        if not isinstance(shared, dict):
            raise RuntimeError("Expected `shared` to be a dictionary, got `{}` instead.".format(type(shared)))

        shared = copy.deepcopy(shared)

        require_base = []
        required_as_base = {}

        for config_id, config in shared.items():
            if "based_on" in config:
                require_base.append(config_id)
                required_id = config["based_on"]
                if required_id not in shared:
                    raise RuntimeError("Config `{}` is based on missing config `{}`.".format(config_id, required_id))

                if required_id not in required_as_base:
                    required_as_base[required_id] = 0
                required_as_base[required_id] += 1

        while len(require_base) > 0:
            required_and_resolved = [required_id for required_id in required_as_base if
                                     (required_id not in require_base)]
            if len(required_and_resolved) == 0:
                raise RuntimeError("Could not resolve configs, cyclic dependency detected between the "
                                   "following shared configs: {}".format(require_base))

            for config_id in require_base:
                unresolved_config = shared[config_id]
                if unresolved_config["based_on"] not in required_and_resolved:
                    continue

                # noinspection PyTypeChecker
                based_on: str = unresolved_config.pop("based_on")
                base_config = shared[based_on]
                resolved_config = {**base_config, **unresolved_config}

                shared[config_id] = resolved_config
                require_base.remove(config_id)
                required_as_base[based_on] -= 1
                if required_as_base[based_on] == 0:
                    required_as_base.pop(based_on)

        return shared

    def run(self) -> None:
        datasets: dict[str, AbstractProtoSet] = get_proto_datasets(self.config["datasets"])

        print("Starting a series of experiments (logs: {})".format(self.log_dir))
        for experiment_name, experiment_config in self.experiment_configs.items():
            experiment_config = copy.deepcopy(experiment_config)
            # region Checking "skip" tag
            skip = experiment_config.pop("skip") if "skip" in experiment_config else False
            if skip == "yes":
                print("Skipping experiment `{}` ('skip' = '{}').".format(experiment_name, skip))
                continue
            # endregion

            print("Starting experiment `{}`...".format(experiment_name))

            # region Add base config (if needed)
            if "based_on" in experiment_config:
                # noinspection PyTypeChecker
                base_config_id: str | None = experiment_config.pop("based_on")
            elif "default" in self.shared_configs:
                base_config_id = "default"
            else:
                base_config_id = None

            if base_config_id is not None:
                base_config = self.shared_configs[base_config_id]
                experiment_config = {
                    **base_config,
                    **experiment_config
                }
            # endregion

            # region Pop `dataset`, `seeds` and `folds` from config
            # noinspection PyTypeChecker
            dataset_name: str = experiment_config.pop("dataset")
            seeds: list[int] | None
            if "seeds" in experiment_config:
                # noinspection PyTypeChecker
                seeds = experiment_config.pop("seeds")
            else:
                seeds = None
            # noinspection PyTypeChecker
            folds_config: str | int | list[int] = experiment_config.pop("folds")
            # endregion

            # region Get folds for experiment
            dataset = datasets[dataset_name]
            folds = dataset.get_experiment_folds(folds_config)
            # endregion

            # region Add dataset and log_dir config
            experiment_log_dir = self.log_dir / experiment_name
            additional_config = {
                "scalar_features_path": dataset.scalar_features_path,
                "categorical_features_path": dataset.categorical_features_path,
                "preparation_pipeline_export": dataset.preparation_pipeline_export_path,
                "log_dir": experiment_log_dir,
            }

            experiment_config.update(additional_config)
            # endregion

            # region Resolve checkpoint(s) if present
            experiment_config = self.resolve_checkpoints(experiment_config)
            if "checkpoint" in experiment_config:
                checkpoint = experiment_config.pop("checkpoint")
            else:
                checkpoint = None
            # endregion

            experiment = Experiment(experiment_config, folds, seeds, checkpoint=checkpoint, name=experiment_name)
            experiment.run()

            if experiment.has_test_stage:
                gather_test_results(root=self.log_dir, summarize_folds=True, save_absolute_path=False)
                print("Gathered test results in `{}`...".format(self.log_dir))

            print("Completed experiment `{}`.".format(experiment_name))
        print("Successfully completed this series of experiments! (logs: {})".format(self.log_dir))

    def resolve_checkpoints(self, config: dict[str, Any]) -> dict[str, Any]:
        """
            Updates the checkpoints in the config and sub-config recursively.
        """
        for key in config:
            if key == "checkpoint":
                checkpoint: str | Path = config[key]

                if checkpoint.startswith("$"):
                    checkpoint = checkpoint[1:]

                if checkpoint in self.experiment_configs:
                    checkpoint = self.log_dir / checkpoint
                else:
                    checkpoint = Path(checkpoint)

                if checkpoint.name != "lightning_logs":
                    lightning_dir = checkpoint / "lightning_logs"
                    if lightning_dir.exists():
                        checkpoint = lightning_dir

                config[key] = checkpoint
            
            elif isinstance(config[key], dict):
                config[key] = self.resolve_checkpoints(config[key])

        return config


    @property
    def experiment_configs(self) -> dict[str, ExperimentConfig]:
        return self.config["experiments"]

    @property
    def shared_configs(self) -> dict[str, ExperimentConfig]:
        return self.config["shared"]

    @property
    def dataset_configs(self) -> dict[str, DatasetConfig]:
        return self.config["datasets"]
