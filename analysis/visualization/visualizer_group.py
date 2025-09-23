import torch
from pathlib import Path
from tqdm import tqdm
from typing import Any, Sequence, Iterator, Generator

from data import ModalitySet, Modality, Sample
from models.module import MindfulModule, ModelInput
from models.classification.abstract_classifier import AbstractClassifier, PrototypeLayer
from models.classification.monai_classifier import MonaiClassifier
from models.segmentation.segmentation_unet import SegmentationUNet
from analysis.visualization import (
    Visualizer, 
    SaliencyMap, 
    CAMs, 
    OcclusionMap, 
    AttentionMap, 
    AttentionRecorder, 
    SegmentationMap,
    MatchSamples)
from utils.misc import load_json

DataLoaderOutput = tuple[list[Sample], ModelInput]


class VisualizerGroup(object):
    def __init__(self,
                 visualizer_configs: dict[str, dict[str, Any]] | str | Path,
                 model: MindfulModule,
                 log_dir: str | Path,
                 input_modalities: ModalitySet,
                 features_names: dict[Modality, list[str]] = None,
                 ):
        if isinstance(visualizer_configs, (str, Path)):
            visualizer_configs = load_json(visualizer_configs)

        self.visualizer_configs = visualizer_configs
        self.model = model
        self.log_dir = log_dir
        self.input_modalities = input_modalities
        self.features_names = features_names
        self.visualizers = [self.make_visualizer(visualizer, visualizer_config)
                            for visualizer, visualizer_config in visualizer_configs.items()
                            if not visualizer.startswith("-")]

    # region Run
    def __call__(self,
                 sample: Sequence[torch.Tensor],
                 **kwargs
                 ) -> dict[Visualizer, dict[str, torch.Tensor]]:
        return {visualizer: visualizer(sample, **kwargs) for visualizer in self.visualizers}
    
    def run(self, data_loader: Iterator[DataLoaderOutput], finalize: bool = True) -> None:
        for batch, ids in self.wrap_data_loader(data_loader):
            self.run_batch(batch, ids)

        if finalize:
            self.finalize()

    def run_batch(self, batch: list[torch.Tensor], ids: list[str] = None) -> None:
        for visualizer in self.visualizers:
            visualizer.run_batch(batch, ids=ids)

    def finalize(self) -> None:
        for visualizer in self.visualizers:
            visualizer.finalize()
    # endregion

    # region Dataloader utils
    def wrap_data_loader(self, data_loader: Iterator[DataLoaderOutput]
                         ) -> Generator[tuple[ModelInput, list[str]], None, None]:
        for samples, batch in data_loader:
            ids = self.samples_to_ids(samples)
            batch = self.to_model_device(batch)
            yield batch, ids

    def to_model_device(self, batch: ModelInput) -> ModelInput:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.model.device)
        
        return [x.to(self.model.device) for x in batch]

    @staticmethod
    def samples_to_ids(samples: list[Sample]) -> list[str]:
        return [sample.id for sample in samples]
    # endregion

    # region Fit
    def fit(self, 
            train_data_loader: Iterator[DataLoaderOutput] | None, 
            validation_data_loader: Iterator[DataLoaderOutput] | None,
            ) -> None:
        if not self.should_fit:
            return
        
        if self.should_fit_with_train_data:
            if train_data_loader is None:
                raise RuntimeError("Fitting on training data is required but no train data loader was provided.")
            self.fit_with_train_data(train_data_loader)

        if self.should_fit_with_validation_data:
            if validation_data_loader is None:
                raise RuntimeError("Fitting on validation data is required but no validation data loader was provided.")
            self.fit_with_validation_data(validation_data_loader)

        self.on_fit_end()

    # region Dataloader level
    def fit_with_train_data(self, data_loader: Iterator[DataLoaderOutput]):
        for batch, ids in tqdm(self.wrap_data_loader(data_loader), desc="Fitting visualizers on `train` data"):
            self.fit_with_train_batch(batch, ids)

    def fit_with_validation_data(self, data_loader: Iterator[DataLoaderOutput]):
        for batch, ids in tqdm(self.wrap_data_loader(data_loader), desc="Fitting visualizers on `validation` data"):
            self.fit_with_validation_batch(batch, ids)

    # endregion

    # region Batch level
    def fit_with_train_batch(self, batch: ModelInput, ids: list[str] = None) -> None:
        for visualizer in self.visualizers:
            if visualizer.should_fit_with_train_data:
                visualizer.fit_with_train_batch(batch, ids)
    
    def fit_with_validation_batch(self, batch: ModelInput, ids: list[str] = None) -> None:
        for visualizer in self.visualizers:
            if visualizer.should_fit_with_validation_data:
                visualizer.fit_with_validation_batch(batch, ids)
    # endregion

    def on_fit_end(self) -> None:
        for visualizer in self.visualizers:
            if visualizer.should_fit:
                visualizer.on_fit_end()

    @property
    def should_fit(self) -> bool:
        return any([visualizer.should_fit for visualizer in self.visualizers])

    @property
    def should_fit_with_train_data(self) -> bool:
        return any([visualizer.should_fit_with_train_data for visualizer in self.visualizers])
    
    @property
    def should_fit_with_validation_data(self) -> bool:
        return any([visualizer.should_fit_with_validation_data for visualizer in self.visualizers])

    # endregion

    # region Instantiate
    def make_visualizer(self, visualizer: str, visualizer_config: dict[str, Any]) -> Visualizer:
        config = {
            "model": self.model,
            "log_dir": self.log_dir,
            "input_modalities": self.input_modalities,
            "features_names": self.features_names
        }
        config.update(visualizer_config)

        visualizer = visualizer.lower()
        if "saliency" in visualizer:
            return SaliencyMap(**config)
        elif "occlusion" in visualizer:
            return OcclusionMap(**config)
        elif "attention" in visualizer:
            return AttentionMap(**config)
        elif ("cam" in visualizer) or ("class_activation_map" in visualizer):
            return CAMs(**config)
        elif "segment" in visualizer:
            return SegmentationMap(**config)
        elif "match" in visualizer:
            return MatchSamples(**config)
        else:
            raise ValueError(visualizer)

    @staticmethod
    def get_default_visualizers_config(model: MindfulModule) -> dict[str, dict[str, Any]]:
        visualizer_configs = {}

        if isinstance(model, MonaiClassifier):
            visualizer_configs["class_activation_map"] = {}

        if not isinstance(model, SegmentationUNet):
            attention_modules = AttentionRecorder.find_attention_modules(model)
            if len(attention_modules) > 0:
                visualizer_configs["attention"] = {
                    "saved_versions": ["next_to_overlay", "next_to_additive", "next_to_hsv", "next_to_mul"],
                }

        if isinstance(model, AbstractClassifier):
            visualizer_configs["saliency"] = {
                "smooth_maps": True
            }

            visualizer_configs["occlusion"] = {
                "image_kernel_size": 16,
                "image_kernel_overlap": 0.6,
                "image_kernel_sigma": 0.25
            }

            if PrototypeLayer.model_has_prototype_layer(model):
                visualizer_configs["match_samples"] = {
                    "k": 1
                }

        if isinstance(model, SegmentationUNet):
            visualizer_configs["segmentation_maps"] = {

            }

        return visualizer_configs

    # endregion
