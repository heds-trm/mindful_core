import numpy as np
import torch
from monai.transforms import SaveImage
from monai.data import MetaTensor
import pickle
import copy
from pathlib import Path
import warnings
from typing import Any

from models.classification import AbstractClassifier
from models.index import get_model, ModuleConfig
from data.transforms.pipeline import Pipeline
from data.modalities import ModalityType, Modality
from utils.misc import load_json

INPUT_SAMPLE_DATA = tuple[MetaTensor, ...] | MetaTensor
INPUT_BATCH_DATA = list[INPUT_SAMPLE_DATA]

INPUT_SAMPLE_METADATA = tuple[dict[str, Any], ...]
INPUT_BATCH_METADATA = list[INPUT_SAMPLE_METADATA]


# region Restoration (model, pipelines)
def restore_model(metadata_path: Path, checkpoint_path: Path) -> tuple[AbstractClassifier, Pipeline | None, Pipeline]:
    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)

    load_config: ModuleConfig = metadata["model_config"]
    # noinspection PyTypeChecker
    model: AbstractClassifier = get_model(load_config, checkpoint=checkpoint_path)
    preparation_pipeline = metadata.get("preparation_pipeline")
    preproc_pipeline = metadata["preproc_pipeline"]

    if "model_thresholds" in metadata:
        for threshold_name, threshold in metadata["model_thresholds"].items():
            if hasattr(model.classification_summary, threshold_name):
                setattr(model.classification_summary, threshold_name, threshold)
    else:
        warnings.warn("You are using an old metadata file that is missing model thresholds.")

    return model, preparation_pipeline, preproc_pipeline


# endregion

# region Sample preparation / loading / preprocessing
def find_json_samples(input_paths: list[Path]) -> list[Path]:
    sample_paths = []
    for path in input_paths:
        if not path.exists():
            raise RuntimeError(path)

        if path.is_dir():
            for sub_path in path.iterdir():
                if sub_path.is_file() and sub_path.suffix == ".json":
                    sample_paths.append(sub_path)
        else:
            sample_paths.append(path)

    return sample_paths


def get_image_savers(pipeline: Pipeline, output_dir: str, output_ext: str = ".mha") -> dict[Modality, SaveImage]:
    image_modalities = [modality for modality in pipeline.output_modalities
                        if modality.type == ModalityType.IMAGE]
    image_savers = {
        modality:
            SaveImage(output_dir=output_dir,
                      output_ext=output_ext,
                      output_postfix="",
                      resample=False,
                      separate_folder=False,
                      print_log=False)
        for modality in image_modalities
    }

    return image_savers


def prepare_sample(sample_data: dict,
                   preparation_pipeline: Pipeline,
                   image_savers: dict[Modality, SaveImage] = None
                   ) -> tuple[dict[str, Any], dict]:
    sample_data = copy.copy(sample_data)
    data_to_prepare = {modality_name: value for modality_name, value in sample_data.items()
                       if modality_name in preparation_pipeline.input_modalities}

    prepared_data = preparation_pipeline(data_to_prepare)
    if isinstance(prepared_data, MetaTensor):
        prepared_data = (prepared_data,)

    meta_data = {}
    input_image_modality = preparation_pipeline.input_modalities.get(ModalityType.IMAGE)
    input_image_modality_id = input_image_modality.id if input_image_modality is not None else None
    for modality, value in zip(preparation_pipeline.output_modalities, prepared_data):
        modality: Modality
        value: MetaTensor | np.ndarray

        meta_data[modality.id] = value.meta
        if modality.type == ModalityType.IMAGE:
            saver = image_savers[modality]

            input_filepath = Path(sample_data[input_image_modality_id])
            # noinspection PyUnresolvedReferences
            output_folder = Path(saver.folder_layout.output_dir)

            base_filename_stem = "{}_{}".format(input_filepath.stem, modality.id)
            base_filename = base_filename_stem + saver.output_ext

            output_filepath_stem = output_folder / base_filename_stem
            output_filepath = output_folder / base_filename

            saver(value, filename=output_filepath_stem)
            sample_data[modality.id] = output_filepath.as_posix()
        else:
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            sample_data[modality.id] = value.tolist()

    return sample_data, meta_data


def load_sample(inputs: str | Path | dict[str, Any],
                preproc_pipeline: Pipeline,
                preparation_pipeline: Pipeline = None,
                preparation_savers: dict[Modality, SaveImage] = None,
                to_gpu: bool = True
                ) -> INPUT_SAMPLE_DATA:
    input_data = load_json(inputs) if isinstance(inputs, (Path, str)) else inputs

    if preparation_pipeline is not None:
        input_data, meta_data = prepare_sample(input_data, preparation_pipeline, preparation_savers)
    else:
        meta_data = {}

    input_data = {modality_name: value for modality_name, value in input_data.items()
                  if modality_name in preproc_pipeline.input_modalities}

    if "label" in preproc_pipeline.input_modalities:
        input_data["label"] = -1

    with preproc_pipeline.disable_modalities(["label"]):
        sample_data = preproc_pipeline(input_data)

    if isinstance(sample_data, torch.Tensor):
        sample_data = (sample_data,)

    if to_gpu:
        sample_data = tuple(modality_value.cuda() for modality_value in sample_data)

    model_input_modalities = preproc_pipeline.output_modalities - ModalityType.LABEL
    for modality, modality_value in zip(model_input_modalities, sample_data):
        modality: Modality
        if not isinstance(modality_value, MetaTensor):
            raise RuntimeError(type(modality_value))

        if (modality.id not in meta_data) or ("transforms" not in meta_data[modality.id]):
            continue

        if "transforms" not in modality_value.meta:
            modality_value.meta["transforms"] = meta_data[modality.id]["transforms"]
        else:
            modality_value.meta["transforms"] = {**meta_data[modality.id]["transforms"],
                                                 **modality_value.meta["transforms"]}

    if len(sample_data) == 1:
        sample_data = sample_data[0]

    return sample_data


def load_samples(sample_paths: list[Path],
                 preproc_pipeline: Pipeline,
                 preparation_pipeline: Pipeline = None,
                 preparation_savers: dict[Modality, SaveImage] = None,
                 to_gpu=True,
                 ) -> tuple[INPUT_BATCH_DATA, INPUT_BATCH_METADATA]:
    samples: INPUT_BATCH_DATA = []
    for path in sample_paths:
        sample_data, original_input_data = load_sample(path, preproc_pipeline, preparation_pipeline, preparation_savers,
                                                       to_gpu=to_gpu)
        samples.append(sample_data)

    meta_data: list[tuple[dict[str, Any], ...]] = [
        tuple(tensor.meta for tensor in sample)
        for sample in samples
    ]

    return samples, meta_data


def stack_samples(samples: list[tuple[MetaTensor, ...]]):
    modality_count = len(samples[0])
    samples = tuple(
        torch.stack([sample[i] for sample in samples], dim=0)
        for i in range(modality_count)
    )
    return samples

# endregion
