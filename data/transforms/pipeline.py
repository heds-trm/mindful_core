import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import ToTensor, Transform, MapTransform, Randomizable
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import MAX_SEED, get_seed
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
from enum import IntEnum
import warnings
from pathlib import Path
from typing import Any, TypedDict, Optional

from mindful_core.data import ModalitySet, Modality, ModalityType, Sample
from mindful_core.data.transforms.batch_transform import BatchTransform
from mindful_core.data.transforms.serializable_transform import SerializableTransform, TransformParameters
from mindful_core.utils.misc import write_json, try_load_json, DynValue, ContextValue

"""
{
    "inputs":
    {
        "id": "type"
    },
    "outputs":
    {
        "id": "type"
    },
    }
    "stages":
    {
        "stage":
        [
            {
                "modalities": ["id_1", "id_2"],
                "type": "transform_type",
                "parameters: {}
            }
        ]
    }
}
"""

# region Typing definition
# ModalityConfig may evolve later to allow for additional parameters
ModalityConfig = dict[str, str]


class TransformConfig(TypedDict):
    modalities: list[str] | None
    outputs: list[str] | None
    type: str
    parameters: TransformParameters | None


StageConfig = list[TransformConfig]


class PipelineConfig(TypedDict):
    inputs: dict[str, str]
    outputs: dict[str, str] | None
    stages: dict[str, StageConfig]


TransformInput = dict[str, Any]
TransformOutput = dict[str, MetaTensor]
IntermediateSampleModality = tuple[NdarrayOrTensor, ...] | list[NdarrayOrTensor] | NdarrayOrTensor
ViewOutput = MetaTensor | tuple[MetaTensor, ...]
PipelineOutput = ViewOutput | tuple[ViewOutput, ...]


# endregion


# region Stages
class StageID(IntEnum):
    PREPROCESS = 0
    SHARED_AUGMENT = 1
    VIEW_AUGMENT = 2
    TRAIN_ONLY = 3
    VALIDATION_ONLY = 4
    TEST_ONLY = 5

    def to_key(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_key(key: str) -> Optional["StageID"]:
        key = key.lower()
        for flag in StageID:
            if flag.to_key() == key:
                return flag
        return None

    @property
    def in_train(self) -> bool:
        return self in [StageID.PREPROCESS,
                        StageID.SHARED_AUGMENT,
                        StageID.VIEW_AUGMENT,
                        StageID.TRAIN_ONLY]

    @property
    def in_validation(self) -> bool:
        return self in [StageID.PREPROCESS,
                        StageID.SHARED_AUGMENT,
                        StageID.VIEW_AUGMENT,
                        StageID.VALIDATION_ONLY]

    @property
    def in_test(self) -> bool:
        return self in [StageID.PREPROCESS, StageID.TEST_ONLY]


class KeyedTransform(object):
    def __init__(self,
                 keys: list[str],
                 transform: Transform | SerializableTransform,
                 outputs: list[str] = None
                 ):
        for key in keys:
            if key.startswith("-"):
                raise ValueError("Modality `{}` is ignored.".format(key))
        self.keys: list[str] = keys
        self.transform: Transform | SerializableTransform = transform
        self.outputs: list[str] = outputs or keys

    def __call__(self, sample: TransformInput) -> TransformOutput:
        self.check_input_modalities(sample)

        if isinstance(self.transform, SerializableTransform):
            sample = [sample[key] for key in self.keys]
            sample = self.transform(*sample)

            if not isinstance(sample, dict):
                if len(self.outputs) == 1:
                    sample = [sample]
                sample = {key: value for key, value in zip(self.outputs, sample)}

        elif isinstance(self.transform, MapTransform):
            sample = {key: sample[key] for key in self.keys}
            sample = self.transform(sample)
            if isinstance(sample, (list, tuple)):
                sample = sample[0]

        else:
            sample = {out_key: self.transform(sample[in_key]) for in_key, out_key in zip(self.keys, self.outputs)}

        sample = self.add_output_metadata(sample)
        return sample

    def add_output_metadata(self, sample: TransformInput) -> TransformOutput:
        for modality_id in self.outputs:
            if modality_id not in sample:
                missing_keys = [modality_id for modality_id in self.outputs if modality_id not in sample]
                raise ValueError("Expected sample to contain all outputs keys."
                                 " Missing keys: `{}` for transform of type {}.".
                                 format(missing_keys, type(self.transform)))

            value = sample[modality_id]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            if not isinstance(value, MetaTensor):
                value = MetaTensor(value)
            value.meta["modality"] = modality_id

        return sample

    # region Fit on preprocessed data for SerializableTransform(s)
    def reset_preprocessed(self) -> None:
        if not self.requires_fitting_preprocessed:
            return

        self.transform.reset_preprocessed()

    def fit_preprocessed(self, sample: TransformInput) -> None:
        if not self.requires_fitting_preprocessed:
            return

        sample = [sample[key] for key in self.keys]
        self.transform.fit_preprocessed(*sample)

    def aggregate_preprocessed(self) -> None:
        if not self.requires_fitting_preprocessed:
            return

        self.transform.aggregate_preprocessed()

    @property
    def requires_fitting_preprocessed(self) -> bool:
        return (isinstance(self.transform, SerializableTransform) and
                self.transform.requires_fitting_preprocessed)

    # endregion

    @property
    def name(self) -> str:
        return type(self.transform).__name__

    # region Sanity checks

    def check_input_modalities(self, sample: TransformInput):
        if not isinstance(sample, dict):
            raise TypeError("Sample must be a dictionary, got a `{}`.".format(type(sample)))

        wrong_key_types = [key for key in sample if not isinstance(key, str)]
        if len(wrong_key_types) > 0:
            wrong_key_def = {key: type(key) for key in wrong_key_types}
            raise TypeError("All keys must be strings, got {}.".format(wrong_key_def))

        missing_keys = [key for key in self.keys if key not in sample]
        if len(missing_keys) > 0:
            raise ValueError("The following keys are missing for transform `{}` : {}."
                             " Got {}.".format(self.name, missing_keys, list(sample.keys())))

    # endregion


class Stage(object):
    def __init__(self, config: StageConfig, ignored_modalities: list[str] = None):
        ignored_modalities = ignored_modalities or []
        self.transforms: list[KeyedTransform] = []

        for transform_config in config:
            modalities = transform_config["modalities"]
            if self.ignore_transform(modalities, ignored_modalities):
                continue

            transform = SerializableTransform.deserialize(transform_config["type"], transform_config["parameters"],
                                                          modalities)
            outputs = transform_config.get("outputs") or SerializableTransform.get_default_outputs(transform)
            self.transforms.append(KeyedTransform(modalities, transform, outputs))

    def __call__(self, sample: TransformInput) -> TransformOutput:
        for transform in self.transforms:
            modalities = transform(sample)
            sample.update(modalities)
        return sample

    def serialize(self) -> StageConfig:
        stage_config: list[TransformConfig] = [
            {
                "modalities": keyed_transform.keys,
                "type": str(type(keyed_transform.transform)),
                "parameters": SerializableTransform.serialize(keyed_transform.transform),
                "outputs": keyed_transform.outputs,
            }
            for keyed_transform in self.transforms
        ]

        return stage_config

    @staticmethod
    def ignore_transform(transform_modalities: list[str], ignored_modalities: list[str]) -> bool:
        return any([(modality.startswith("-") or (modality in ignored_modalities))
                    for modality in transform_modalities])

    def remove_input_modality(self, modality: str):
        self.transforms = [transform for transform in self.transforms
                           if modality not in transform.keys]


# endregion

class Pipeline(Randomizable):
    _disabled_modalities = DynValue([])

    def __init__(self,
                 config: PipelineConfig | Path | str,
                 multiview: bool | int,
                 preprocess_device: str | None = None,
                 ):
        if isinstance(config, (str, Path)):
            config: PipelineConfig = try_load_json(config, "Pipeline config")

        # may need to differentiate inputs modalities from output modalities (for dynamic multimodal masking/weighting)
        input_modalities = Pipeline.get_config_modalities(config["inputs"])
        output_modalities = (Pipeline.get_config_modalities(config["outputs"])
                             if "outputs" in config else input_modalities)
        self.input_modalities: ModalitySet = input_modalities.used_modalities
        self.output_modalities: ModalitySet = output_modalities.used_modalities
        self.stages: dict[StageID, Stage] = {
            StageID.from_key(stage_id): Stage(stage_config, input_modalities.ignored_modalities.ids)
            for stage_id, stage_config in config["stages"].items()}

        self._flat_transforms = self.get_transforms_flattened()
        self.batch_transforms = [transform for transform in self._flat_transforms
                                 if isinstance(transform, BatchTransform)]

        self.preprocess_device = preprocess_device
        self.to_tensor = ToTensor(device=preprocess_device)
        self.multiview = multiview
        self._enable_augmentations = DynValue(True)

        self._fit_samples_ids = []
        self.set_random_state(seed=get_seed())

    # region Initialisation
    @staticmethod
    def get_config_modalities(config: dict[str, str]) -> ModalitySet:
        modalities = [Modality(modality_type, modality_id)
                      for modality_id, modality_type in config.items()]

        return ModalitySet(modalities)

    # endregion

    # region Randomizable
    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> "Pipeline":
        super().set_random_state(seed=seed, state=state)
        for transform in self._flat_transforms:
            if not isinstance(transform, Randomizable):
                continue
            transform.set_random_state(seed=self.R.randint(MAX_SEED, dtype="uint32"))
        return self

    def randomize(self, data: Any | None = None) -> None:
        for transform in self._flat_transforms:
            if not isinstance(transform, Randomizable):
                continue
            try:
                transform.randomize(data)
            except TypeError as type_error:
                tfm_name: str = type(transform).__name__
                warnings.warn("Transform \"{tfm_name}\" in Compose not randomized\n{tfm_name}.{type_error}."
                              .format(tfm_name=tfm_name, type_error=type_error), RuntimeWarning)

    # endregion

    # region Conversion (no labels, to validation, ...)
    @staticmethod
    def remove_labels_from_config(config: PipelineConfig):
        config: PipelineConfig = deepcopy(config)

        label_key = ModalityType.LABEL.to_key()
        input_label_modalities = [key for key, value in config["inputs"].items() if value == label_key]
        output_label_modalities = [key for key, value in config["outputs"].items() if value == label_key]

        for input_label_modality in input_label_modalities:
            config["inputs"].pop(input_label_modality)

        for output_label_modality in output_label_modalities:
            config["outputs"].pop(output_label_modality)

        for stage_id, stage_config in config["stages"].items():
            transforms_to_remove = []
            for transform_config in stage_config:
                if any([input_label_modality in transform_config["modalities"]
                        for input_label_modality in input_label_modalities]):
                    transforms_to_remove.append(transform_config)
                elif (("outputs" in transform_config) and
                      any([output_label_modality in transform_config["outputs"]
                           for output_label_modality in output_label_modalities])):
                    transforms_to_remove.append(transform_config)

            for transform_to_remove in transforms_to_remove:
                stage_config.remove(transform_to_remove)

        return config

    @staticmethod
    def convert_to_train_config(config: PipelineConfig) -> PipelineConfig:
        config = deepcopy(config)

        config["stages"] = {
            stage_id: stage_config
            for stage_id, stage_config in config["stages"].items()
            if StageID.from_key(stage_id).in_train
        }

        return config

    @staticmethod
    def convert_to_val_config(config: PipelineConfig) -> PipelineConfig:
        config = deepcopy(config)

        config["stages"] = {
            stage_id: stage_config
            for stage_id, stage_config in config["stages"].items()
            if StageID.from_key(stage_id).in_validation
        }

        return config

    @staticmethod
    def convert_to_test_config(config: PipelineConfig) -> PipelineConfig:
        config = deepcopy(config)

        config["stages"] = {
            stage_id: stage_config
            for stage_id, stage_config in config["stages"].items()
            if StageID.from_key(stage_id).in_test
        }

        return config

    # endregion

    # region Fit/Run
    def is_fit(self, samples: list[Sample] = None) -> bool:
        if len(self._fit_samples_ids) == 0:
            return False

        if samples is None:
            return len(self._fit_samples_ids) > 0
        else:
            if len(self._fit_samples_ids) != len(samples):
                return False

            return all([sample.id in self._fit_samples_ids for sample in samples])

    def fit(self, samples: list[Sample]) -> None:
        if self.is_fit(samples):
            return

        self.fit_raw(samples)
        self.fit_preprocessed(samples)

        self._fit_samples_ids = [sample.id for sample in samples]

    def fit_raw(self, samples: list[Sample]) -> None:
        transforms_to_fit = SerializableTransform.get_transforms_to_fit(self._flat_transforms)
        for transform in transforms_to_fit:
            transform.fit(samples)

    def fit_preprocessed(self, samples: list[Sample]) -> None:
        """
        This method runs up to every transform that fit on already pre-processed data. \n
        This allows for transforms to fit on the same (training) data as they'll process later. \n
        For example, this allows for extracting the mean and std of images, which would not be
           possible only using the filepath and not knowing previous pre-processes.
        """

        samples: list[TransformInput] = [sample.get_values(self.input_modalities) for sample in samples]
        transforms = self.get_transforms_flattened(keyed=True)

        transforms_to_fit = [transform for transform in transforms if transform.requires_fitting_preprocessed]
        if len(transforms_to_fit) == 0:
            return

        # TODO: Add option to reload data between chunks

        transforms_chunks: list[list[KeyedTransform]] = []
        chunk: list[KeyedTransform] = []
        for transform in transforms:
            transform: KeyedTransform
            chunk.append(transform)

            if transform.requires_fitting_preprocessed:
                transforms_chunks.append(chunk)
                chunk = []

        for chunk in transforms_chunks:
            transform_to_fit: KeyedTransform = chunk[-1]
            description = "Fitting transform `{}`".format(transform_to_fit.name)

            transform_to_fit.reset_preprocessed()
            for i in tqdm(range(len(samples)), desc=description):
                sample = samples[i]
                for transform in chunk[:-1]:
                    update = transform(sample)
                    sample.update(update)
                transform_to_fit.fit_preprocessed(sample)
                if transform_to_fit is transforms_to_fit[-1]:
                    # Free memory 
                    # noinspection PyTypeChecker
                    samples[i] = None
                    del sample
            transform_to_fit.aggregate_preprocessed()

            if transform_to_fit is not transforms_to_fit[-1]:
                for sample in samples:
                    update = transform_to_fit(sample)
                    sample.update(update)

    def __call__(self, sample: Sample | TransformInput) -> PipelineOutput:
        if isinstance(sample, Sample):
            sample = sample.get_values(self.input_modalities)

        sample = self.preprocess(sample)

        if self.enable_augmentations:
            sample = self.augment_and_finalize(sample)
        else:
            sample = self.finalize(sample)

        return sample

    # region Run stage (Preprocessors, Shared augmentations, View augmentations)
    def run_stage(self, sample: TransformInput, stage_id: StageID) -> TransformOutput:
        if stage_id not in self.stages:
            return sample
        return self.stages[stage_id](sample)

    def run_stages(self, sample: TransformInput, *stage_ids: StageID) -> TransformOutput:
        for stage_id in stage_ids:
            sample = self.run_stage(sample, stage_id)
        return sample

    def preprocess(self, sample: TransformInput) -> TransformOutput:
        return self.run_stage(sample, StageID.PREPROCESS)

    def augment_and_finalize(self, sample: TransformInput) -> PipelineOutput:
        sample = self.shared_augment(sample)

        if self.use_multiview:
            sample = tuple([self.finalize(self.view_augment(sample)) for _ in range(self.view_count)])
        else:
            sample = self.finalize(self.view_augment(sample))
        return sample

    def shared_augment(self, sample: TransformInput) -> TransformOutput:
        return self.run_stage(sample, StageID.SHARED_AUGMENT)

    def view_augment(self, sample: TransformInput) -> TransformOutput:
        return self.run_stage(sample, StageID.VIEW_AUGMENT)

    # endregion

    # region Finalize
    def finalize(self, sample: dict[str, NdarrayOrTensor]) -> ViewOutput:
        sample = self.run_stages(sample, StageID.VALIDATION_ONLY, StageID.TEST_ONLY)
        sample = [self.finalize_modality(sample[modality.id])
                  for modality in self.output_modalities
                  if self.modality_allowed(modality)]

        if len(sample) == 1:
            return sample[0]
        else:
            return tuple(sample)

    def finalize_modality(self,
                          sample_modality: IntermediateSampleModality
                          ) -> tuple[MetaTensor, ...] | MetaTensor:
        if isinstance(sample_modality, (tuple, list)):
            sample_modality = tuple([self.to_tensor(x) for x in sample_modality])
        else:
            sample_modality = self.to_tensor(sample_modality)
        return sample_modality

    def modality_allowed(self, modality: Modality) -> bool:
        return ((modality.type not in self._disabled_modalities) and
                (modality.id not in self._disabled_modalities) and
                (modality not in self._disabled_modalities))

    # endregion
    # endregion

    # region Utility

    def get_transforms_flattened(self,
                                 #  modality_filter: Modality = None,
                                 stage_filter: StageID = None,
                                 keyed=False
                                 ) -> list[Transform] | list[KeyedTransform]:
        transforms = [keyed_transform if keyed else keyed_transform.transform
                      for stage_id, stage in self.stages.items()
                      for keyed_transform in stage.transforms
                      if ((stage_filter is None) or (stage_id == stage_filter))
                      #   and ((modality_filter is None) or (keyed_transform.keys))
                      ]

        return transforms

    @staticmethod
    def disable_modalities(modalities: list[Modality | ModalityType | str] | ModalitySet) -> ContextValue:
        if not isinstance(modalities, (list, tuple, ModalitySet)):
            modalities = [modalities]
        return ContextValue(Pipeline._disabled_modalities, modalities)

    @staticmethod
    def no_labels() -> ContextValue:
        return Pipeline.disable_modalities([ModalityType.LABEL])

    def no_augmentation(self) -> ContextValue:
        return ContextValue(self._enable_augmentations, False)

    def get_modality_type_input_indices(self, modality_type: ModalityType) -> list[int]:
        return self.input_modalities.get_modality_type_indices(modality_type)

    def get_modality_type_output_indices(self, modality_type: ModalityType) -> list[int]:
        return self.output_modalities.get_modality_type_indices(modality_type)

    # endregion

    # region Saving
    def save_config(self, directory: str, prefix: str | None):
        name = prefix + "_pipeline" if prefix else "pipeline"
        filepath = os.path.join(directory, name + ".json")

        input_modalities: dict[str, str] = {
            modality.id: modality.type.to_key()
            for modality in self.input_modalities
        }

        output_modalities: dict[str, str] = {
            modality.id: modality.type.to_key()
            for modality in self.output_modalities
        }

        stages: dict[str, StageConfig] = {
            stage_id.to_key(): stage.serialize()
            for stage_id, stage in self.stages.items()
        }

        config: PipelineConfig = {
            "inputs": input_modalities,
            "outputs": output_modalities,
            "stages": stages
        }

        write_json(filepath, config)

    # endregion

    # region Properties
    @property
    def enable_augmentations(self) -> bool:
        return bool(self._enable_augmentations)

    @property
    def use_multiview(self) -> bool:
        if isinstance(self.multiview, bool):
            return self.multiview
        else:
            return self.multiview > 1

    @property
    def view_count(self) -> int:
        if isinstance(self.multiview, bool):
            return 2 if self.multiview else 1
        else:
            return self.multiview
    # endregion
