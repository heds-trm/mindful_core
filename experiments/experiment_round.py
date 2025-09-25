import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
import pickle
from tqdm import tqdm
from pathlib import Path
import os
import copy
import warnings
from typing import Optional, Iterator

from mindful_core.experiments import ExperimentRoundConfig
from mindful_core.models import MindfulModule
from mindful_core.models.index import get_model, is_representation_model, is_classification_model
from mindful_core.models.model_output import ClassifierOutput
from mindful_core.models.classification import AbstractClassifier
from mindful_core.models.segmentation.segmentation_unet import SegmentationUNet, SegmentationOutput
from mindful_core.analysis.visualization import VisualizerGroup
from mindful_core.analysis.statistics import ConfidenceSummary, logits_to_probabilities, ClassificationSummary
from mindful_core.data import SubsetID, MindfulDataset, Sample, ModalityType
from mindful_core.data.data_folds import DataFold, PresetFold
from mindful_core.data.transforms.pipeline import Pipeline
from mindful_core.utils.callbacks import MilestoneCheckpoint
from mindful_core.utils.parsing import get_monitor_mode


class ExperimentRound(object):
    def __init__(self, config: ExperimentRoundConfig):
        self.config = config
        self._model: MindfulModule | None = None
        self._dataset: MindfulDataset | None = None
        self._pipelines: dict[SubsetID, Pipeline] = {}
        self._trainer: pl.Trainer | None = None
        self._logger: TensorBoardLogger | None = None
        self._model_checkpoint_callbacks: list[ModelCheckpoint] | None = None
        self._multiview: bool | int | None = None

    # region Model building/loading
    def get_model(self,
                  checkpoint: str | None = None,
                  load_now: bool | None = None) -> MindfulModule:
        if (self._model is not None) and (checkpoint is None):
            return self._model

        if load_now is None:
            load_now = not self.config.resume_training

        if self.includes_training and not load_now:
            checkpoint = None
        elif checkpoint is None:
            checkpoint = self.config.checkpoint

        if not load_now:
            checkpoint = None

        model = get_model(module_config=self.config.model, checkpoint=checkpoint)

        self._model = model
        return model

    @property
    def model(self) -> MindfulModule:
        return self.get_model()

    # endregion

    # region MindfulDataset
    @property
    def dataset(self) -> MindfulDataset:
        if self._dataset is None:
            self._dataset = MindfulDataset(fold=self.config.fold,
                                           seed=self.config.seed,
                                           training_data_reduction=self.config.training_data_reduction)
        return self._dataset

    @property
    def pipelines(self) -> dict[SubsetID, Pipeline]:
        if len(self._pipelines) == 0:
            preprocess_device = "cuda" if self.config.accelerator in ["cuda", "gpu"] else "cpu"
            shared_params = {"multiview": self.multiview, "preprocess_device": preprocess_device}

            train_pipeline_config = Pipeline.convert_to_train_config(self.config.base_pipeline_config)
            if is_representation_model(self.config.model_class_name):
                train_pipeline_config = Pipeline.remove_labels_from_config(train_pipeline_config)
                val_pipeline_config = test_pipeline_config = train_pipeline_config
            else:
                val_pipeline_config = Pipeline.convert_to_val_config(self.config.base_pipeline_config)
                test_pipeline_config = Pipeline.convert_to_test_config(self.config.base_pipeline_config)

            self._pipelines[SubsetID.TRAIN] = Pipeline(config=train_pipeline_config, **shared_params)
            self._pipelines[SubsetID.VALIDATION] = Pipeline(config=val_pipeline_config, **shared_params)
            self._pipelines[SubsetID.TEST] = Pipeline(config=test_pipeline_config, **shared_params)

            for pipeline in self._pipelines.values():
                for batch_transform in pipeline.batch_transforms:
                    self.model.register_train_batch_start_hook(batch_transform.on_train_batch_start)
                    self.model.register_train_batch_end_hook(batch_transform.on_train_batch_end)

        return self._pipelines

    def get_data_loaders(self,
                         training: bool,
                         batch_sizes: dict[SubsetID, int | str] | None = None,
                         ) -> dict[SubsetID, DataLoader]:
        if isinstance(self.model, AbstractClassifier):
            if self.config.use_imbalanced_sampler is None:
                use_imbalanced_sampler = True
            else:
                use_imbalanced_sampler = self.config.use_imbalanced_sampler
        else:
            use_imbalanced_sampler = False

        batch_sizes = self.config.batch_sizes if batch_sizes is None else batch_sizes

        return self.dataset.get_data_loaders(
            pipelines=self.pipelines,
            batch_sizes=batch_sizes,
            num_workers=self.config.num_workers,
            use_imbalanced_sampler=use_imbalanced_sampler,
            trainer=self.trainer,
            model=self.model,
            training=training,
        )

    def get_data_loaders_with_batched_samples(self) -> dict[SubsetID, Iterator[tuple]]:
        data_loaders = self.get_data_loaders(training=False)
        samples_iterators = self.dataset.get_batch_samples_iterators(self.config.batch_sizes)
        return {subset_id: zip(samples_iterators[subset_id], data_loaders[subset_id])
                for subset_id in data_loaders}

    # endregion

    # region Train / Test / Visualize / ...
    def run(self):
        self.save_job_metadata()

        if self.includes_training:
            self.train()

        if self.includes_post_training_tasks:
            self.load_best_checkpoint()

        if self.includes_testing:
            self.test()
        elif isinstance(self.model, AbstractClassifier):
            self.compute_inferences()

        if self.includes_pseudo_labelling:
            self.pseudo_label_data()

        if self.includes_confidence_analysis:
            self.confidence_analysis()

        if self.includes_visualization:
            self.visualize()

        if self.includes_representation:
            self.save_representations()

        if self.includes_segmentation:
            self.segment()

        self.wrap_up()

    # region Train
    @property
    def trainer(self) -> pl.Trainer:
        if self._trainer is None:
            self._trainer = pl.Trainer(
                logger=self.logger,
                accelerator=self.config.accelerator,
                devices=self.config.devices,
                default_root_dir=self.config.log_dir,
                detect_anomaly=False,
                callbacks=self.get_callbacks(),
                check_val_every_n_epoch=self.config.validate_every_n_epochs,
                max_epochs=self.config.max_epochs,
                gradient_clip_val=self.config.gradient_clip_value,
                gradient_clip_algorithm=self.config.gradient_clip_algorithm,
            )
        return self._trainer

    def train(self):
        data_loaders = self.get_data_loaders(training=True)

        if self.dataset.should_reduce_training_data:
            reduced_fold = DataFold(seed=self.config.seed, samples=self.dataset.samples)
            reduced_fold.save_scan_data(os.path.join(self.logger.log_dir, "reduced_fold.csv"))

        # self.add_model_metadata_for_logger(data_loaders)
        train_dataloader = data_loaders.get(SubsetID.TRAIN)
        validation_dataloader = data_loaders.get(SubsetID.VALIDATION)

        if train_dataloader is None:
            raise RuntimeError("Cannot train the model: no train dataset available.")

        checkpoint_path = self.config.checkpoint if self.config.resume_training else None
        self.trainer.fit(self.model,
                         train_dataloaders=train_dataloader,
                         val_dataloaders=validation_dataloader,
                         ckpt_path=checkpoint_path)
        self.save_best_model_to_model_dir()

    # endregion

    def load_best_checkpoint(self):
        best_model_path = self._model_checkpoint_callbacks[0].best_model_path

        if not best_model_path:
            raise RuntimeError("Could not determine the best checkpoint based on callbacks. "
                               "Are you missing a pl.ModelCheckpoint callback or "
                               "was this method called before `train()` ?")
        else:
            print("Loading checkpoint from best_model_path `{}`.".format(best_model_path))

        self.get_model(best_model_path, load_now=True)

    def compute_inferences(self, data_loaders: dict[SubsetID, DataLoader] | None = None):
        data_loaders = self.get_data_loaders(training=False) if data_loaders is None else data_loaders

        # region Check and Setup model
        if isinstance(self.model, AbstractClassifier):
            # noinspection PyTypeChecker
            model: AbstractClassifier = self.model
        else:
            raise RuntimeError("Expected model to inherit from AbstractClassifier, got {}".format(type(self.model)))

        if not model.has_classification_thresholds and (SubsetID.VALIDATION in data_loaders):
            val_data_loader = [data_loaders[SubsetID.VALIDATION]]
            model.compute_available_thresholds(self.trainer, val_data_loader, update_thresholds=True)
        # endregion

        inferences_by_subset: dict[SubsetID, pd.DataFrame] = {}
        inferred_features: list[pd.DataFrame] = []
        for subset_id in tqdm(data_loaders, desc="Inferring additional data..."):
            # region Inference
            with Pipeline.no_labels():
                data_loader = data_loaders[subset_id]
                outputs: list[ClassifierOutput] = self.trainer.predict(model, dataloaders=data_loader)

            subset_outputs = ClassifierOutput.concat(outputs)
            subset_indices = [sample.id for sample in self.dataset.samples[subset_id]]
            subset_labels = [sample.label for sample in self.dataset.samples[subset_id]]
            subset_labels = torch.as_tensor(subset_labels, dtype=torch.float32, device="cpu")
            # endregion

            # region Logits / Probabilities
            subset_probabilities = logits_to_probabilities(subset_outputs.logits).cpu()
            subset_inferences = model.classification_summary.get_available_inferences(subset_probabilities,
                                                                                      subset_labels)

            subset_inferences = pd.DataFrame(subset_inferences, index=subset_indices)
            subset_inferences = subset_inferences.assign(SubsetID=subset_id.as_prefix())
            inferences_by_subset[subset_id] = subset_inferences
            # endregion

            # region Features (e.g. individual outputs of ensemble models)
            if model.outputs_features and subset_outputs.has_intermediate_outputs:
                features_names = model.output_features_names
                subset_features = subset_outputs.intermediate_outputs
                if len(subset_features.shape) > 2:
                    batch_size = subset_features.shape[0]
                    subset_features = subset_features.reshape([batch_size, -1])
                subset_features = subset_features.transpose(0, 1).cpu()
                subset_features_dict = {feature_name: {sample_id: float(value)
                                                       for sample_id, value in zip(subset_indices, feature_values)}
                                        for feature_name, feature_values in zip(features_names, subset_features)
                                        }
                subset_features_dataframe = pd.DataFrame(subset_features_dict)
                inferred_features.append(subset_features_dataframe)
            # endregion

        # region Aggregate and Save inferences
        inferences = pd.concat(list(inferences_by_subset.values()))
        inferences_filepath = os.path.join(self.logger.log_dir, "inferences.csv")
        inferences.to_csv(inferences_filepath, index_label="ScanID")

        test_inferences_filepath = os.path.join(self.logger.log_dir, "test_inferences.csv")
        test_inferences = inferences_by_subset[SubsetID.TEST]
        test_inferences.to_csv(test_inferences_filepath, index_label="ScanID")

        if len(inferred_features) > 0:
            inferred_features_frame = pd.concat(inferred_features)
            inferred_features_filepath = os.path.join(self.logger.log_dir, "inferred_features.csv")
            inferred_features_frame.to_csv(inferred_features_filepath, index_label="ScanID")
        # endregion

    def test(self):
        data_loaders = self.get_data_loaders(training=False)

        if (SubsetID.TEST not in data_loaders) or (len(data_loaders[SubsetID.TEST]) == 0):
            raise RuntimeError("No samples provided belonged in the Test subset.")

        if isinstance(self.model, AbstractClassifier):
            self.test_classifier(data_loaders)
        elif isinstance(self.model, SegmentationUNet):
            self.test_segmentation(data_loaders)
        else:
            warnings.warn("No matching Test procedure found for model of type `{}`".format(type(self.model)))

    def test_classifier(self, data_loaders: dict[SubsetID, DataLoader]) -> None:
        if isinstance(self.model, AbstractClassifier):
            # noinspection PyTypeChecker
            model: AbstractClassifier = self.model
        else:
            raise RuntimeError("Expected model to inherit from AbstractClassifier, got {}".format(type(self.model)))

        # region Test results (auroc, accuracy, ...)
        if SubsetID.VALIDATION in data_loaders:
            val_data_loader = [data_loaders[SubsetID.VALIDATION]]
            model.compute_available_thresholds(self.trainer, val_data_loader, update_thresholds=True)

        test_data_loader = [data_loaders[SubsetID.TEST]]
        test_results = self.trainer.test(model, dataloaders=test_data_loader)
        # noinspection PyTypeChecker
        self.save_test_results(test_results)
        # endregion

        self.compute_inferences(data_loaders)

    def test_segmentation(self, data_loaders: dict[SubsetID, DataLoader]) -> None:
        if isinstance(self.model, SegmentationUNet):
            # noinspection PyTypeChecker
            model: SegmentationUNet = self.model
        else:
            raise RuntimeError("Expected model to be a SegmentationUNet, got {}".format(type(self.model)))

        test_data_loader = [data_loaders[SubsetID.TEST]]
        test_results = self.trainer.test(model, dataloaders=test_data_loader)
        # noinspection PyTypeChecker
        self.save_test_results(test_results)

    def segment_roi(self,
                    data_loader: DataLoader,
                    model: SegmentationUNet = None
                    ) -> torch.Tensor:
        if model is None:
            model = self.model

        with Pipeline.disable_modalities(["mask"]):
            outputs: list[SegmentationOutput] = self.trainer.predict(model, dataloaders=data_loader)

        segmentations = SegmentationOutput.concat_segmentations(outputs)
        segmentations = logits_to_probabilities(segmentations, softmax_dim=1)
        return segmentations

    def segment(self) -> None:
        if isinstance(self.model, SegmentationUNet):
            # noinspection PyTypeChecker
            model: SegmentationUNet = self.model
        else:
            raise RuntimeError("Expected model to be a SegmentationUNet, got {}".format(type(self.model)))

        batch_size = 1
        batch_sizes = {SubsetID.TRAIN: 1, SubsetID.VALIDATION: 1, SubsetID.TEST: 1}
        data_loaders = self.get_data_loaders(training=False, batch_sizes=batch_sizes)
        test_data_loader = data_loaders[SubsetID.TEST]

        folder = Path(self.logger.log_dir) / "segmentations"
        folder.mkdir(exist_ok=True)

        from monai.inferers import sliding_window_inference
        roi_size = (64, 64, 64)

        from monai.transforms import SaveImage
        saver = SaveImage(output_dir=folder,
                          output_postfix="seg",
                          output_ext=".mha",
                          separate_folder=False,
                          print_log=False)
        subset_indices = [sample.id for sample in self.dataset.samples[SubsetID.TEST]]
        i = 0

        if self.config.accelerator in ["cuda", "gpu"]:
            self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            for images, masks in test_data_loader:
                outputs = sliding_window_inference(images, roi_size, batch_size, model.backbone)
                outputs = outputs.argmax(dim=1, keepdim=True)
                for image, mask, output in zip(images, masks, outputs):
                    image_name = "image_{}".format(subset_indices[i])
                    mask_name = "mask_{}".format(subset_indices[i])
                    segmentation_name = "segmentation_{}".format(subset_indices[i])
                    saver(image, filename=(folder / image_name).as_posix())
                    saver(mask, filename=(folder / mask_name).as_posix())
                    saver(output, filename=(folder / segmentation_name).as_posix())
                    i += 1
        self.model.train()

    def pseudo_label_data(self) -> None:
        data_loaders = self.get_data_loaders(training=False)
        train_data_loader = [data_loaders[SubsetID.TRAIN]]
        val_data_loader = [data_loaders[SubsetID.VALIDATION]]
        base_fold = self.config.fold

        if not isinstance(self.model, AbstractClassifier):
            raise NotImplementedError("Expected `self.model` to be a (binary) classifier.")

        with Pipeline.no_labels():
            results = self.trainer.predict(self.model, dataloaders=train_data_loader)

        if self.model.yield_confidence:
            predictions = torch.concat([batch[0] for batch in results], dim=0)
            confidence = torch.concat([batch[1] for batch in results], dim=0)
        else:
            predictions = torch.concat(results, dim=0)
            confidence = None

        # region Process current fold based on previous (ref) fold
        if self.config.ref_fold_path is not None:
            ref_fold = PresetFold(self.config.ref_fold_path)

            # Exclusive mode (tmp or keep ?)
            ref_sample_ids = [sample.id for sample in ref_fold.samples[SubsetID.TRAIN]]
            base_train_samples = base_fold.samples[SubsetID.TRAIN]
            redundant_samples = []
            for sample in base_train_samples:
                if sample.id in ref_sample_ids:
                    redundant_samples.append(sample)
            for sample in redundant_samples:
                base_train_samples.remove(sample)
        else:
            print("Warning: you should provide a reference fold, "
                  "e.i. the one originally used for training the model, "
                  "and you should not expect the current fold to contain labels.")
            ref_fold = base_fold
        # endregion

        # region Binary to multiclass
        if self.model.single_class:
            negative_prevalence, _ = ref_fold.get_labels_ratios(ref_fold.samples[SubsetID.TRAIN])
            predictions = self.model.single_to_multiclass(predictions, negative_prevalence)
        # endregion

        if confidence is None:
            raise NotImplementedError("Implement with K folds - for student model selection")
        else:
            self.model.compute_confidence_threshold(self.trainer, val_data_loader, update_threshold=True)
            above_threshold = confidence > self.model.classification_summary.confidence_threshold
            predictions = predictions[above_threshold]

            above_threshold = above_threshold.cpu().numpy()
            base_train_samples = base_fold.samples[SubsetID.TRAIN]
            pseudo_train_samples = [sample for (sample, above) in zip(base_train_samples, above_threshold) if above]

            pseudo_labels = torch.argmax(predictions, dim=-1).cpu().numpy()
            for sample, pseudo_label in zip(pseudo_train_samples, pseudo_labels):
                sample.label = pseudo_label

            pseudo_samples = copy.copy(base_fold.samples)
            pseudo_samples[SubsetID.TRAIN] = pseudo_train_samples

            pseudo_fold = DataFold(samples=pseudo_samples)
            pseudo_fold.save_scan_data(os.path.join(self.logger.log_dir, "pseudo_fold.csv"))

    def confidence_analysis(self) -> None:
        data_loaders = self.get_data_loaders(training=False)
        confidence_summary = ConfidenceSummary(self.model, self.logger.log_dir)
        confidence_summary(data_loaders, self.config.fold)

    def visualize(self) -> None:
        visualizers_config = (self.config.visualizers_config or
                              VisualizerGroup.get_default_visualizers_config(self.model))

        data_loaders = self.get_data_loaders_with_batched_samples()
        train_data_loader = data_loaders.get(SubsetID.TRAIN)
        validation_data_loader = data_loaders.get(SubsetID.VALIDATION)
        test_data_loader = data_loaders.get(SubsetID.TEST)

        if test_data_loader is None:
            raise RuntimeError("No test data loader available to run visualizations.")

        # region Put model in eval mode
        self.model.cuda()
        was_training = self.model.training
        self.model.eval()
        # endregion

        # region Instantiate visualizers
        input_modalities = self.pipelines[SubsetID.TEST].output_modalities
        features_names = {}
        if ModalityType.SCALAR in input_modalities:
            features_names[input_modalities.get(ModalityType.SCALAR)] = self.config.fold.scalar_names

        if "categorical_fields" in visualizers_config:
            categorical_fields: dict[str, list[str]] = visualizers_config.pop("categorical_fields")
            for key in categorical_fields.keys():
                categorical_fields[key].insert(0, "unknown")
        else:
            categorical_fields = self.config.fold.categories_values

        if ModalityType.CATEGORICAL in input_modalities:
            fields_names = ["{}_{}".format(category_name, category_value)
                            for category_name, category_values in categorical_fields.items()
                            for category_value in category_values]
            features_names[input_modalities.get(ModalityType.CATEGORICAL)] = fields_names

        visualizers = VisualizerGroup(visualizer_configs=visualizers_config,
                                      model=self.model,
                                      log_dir=self.logger.log_dir,
                                      input_modalities=input_modalities,
                                      features_names=features_names)
        # endregion

        if visualizers.should_fit:
            visualizers.fit(train_data_loader, validation_data_loader)

        # noinspection PyTypeChecker
        visualizers.run(test_data_loader)

        if was_training:
            self.model.train()

    # region Saving
    def save_representations(self):
        representation_model = self.model
        if hasattr(self.model, "representation_model"):
            representation_model = self.model.representation_model

        if not is_representation_model(representation_model):
            raise RuntimeError("Model class `{}` does not have (or is not) a representation model.".
                               format(type(self.model)))

        data_loaders = self.get_data_loaders_with_batched_samples()

        data_sub_frames: list[pd.DataFrame] = []
        for subset_id, data_loader in data_loaders.items():
            representations = []
            ids = []
            for samples, (batch, _) in data_loader:
                samples: list[Sample]
                batch_ids = [sample.id for sample in samples]
                batch_representations = representation_model(batch)

                ids += batch_ids
                representations.append(batch_representations)
            representations = torch.concat(representations, axis=0).cpu().numpy()
            data_sub_frame = pd.DataFrame(data=representations, index=ids)
            data_sub_frame["SubsetID"] = subset_id.as_prefix()
            data_sub_frames.append(data_sub_frame)

        data_frame = pd.concat(data_sub_frames, axis=0)
        data_frame.index.name = "ScanID"
        data_frame.to_csv(os.path.join(self.logger.log_dir, "representations.csv"))

    def save_job_metadata(self):
        self.init_job_directories()
        self.config.save(self.logger.log_dir)
        self.dataset.save_metadata(self.logger.log_dir)
        for subset_id, pipeline in self.pipelines.items():
            pipeline.save_config(self.logger.log_dir, prefix=subset_id.as_prefix())

    def save_test_results(self, test_results: list[dict[str, float]]):
        if len(test_results) > 1:
            raise NotImplementedError("More than one test result dataset is currently unsupported.")
        test_results = test_results[0]

        test_results_filepath = os.path.join(self.logger.log_dir, "test_results.csv")
        print("Saving test results to {}.".format(test_results_filepath))

        data_frame = pd.DataFrame(data=[list(test_results.values())], columns=list(test_results.keys()))
        data_frame.to_csv(test_results_filepath, index=False)

    def save_best_model_to_model_dir(self):
        # The following model saving is intended for Sagemaker
        if self.config.model_dir is not None:
            save_filename = os.path.join(self.config.model_dir, "model.pth")
            print("==========" * 10)
            print("Training done.\nSaving the model to {}.".format(save_filename))
            print("==========" * 10)
            if not os.path.exists(self.config.model_dir):
                os.makedirs(self.config.model_dir)
            torch.save(self.model.cpu().state_dict(), save_filename)

    def init_job_directories(self):
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        if not os.path.exists(self.logger.log_dir):
            os.makedirs(self.logger.log_dir)

        print("Logging folder: ", self.logger.log_dir)

    def wrap_up(self):
        self.save_inference_components(include_checkpoint=False)

    def save_inference_components(self, include_checkpoint=False):
        if include_checkpoint:
            raise NotImplementedError

        if self.config.preparation_pipeline_export is not None:
            with open(self.config.preparation_pipeline_export, "rb") as file:
                preparation_pipeline = pickle.load(file)
        else:
            preparation_pipeline = None

        preproc_pipeline = self.pipelines[SubsetID.TEST]
        preproc_pipeline.fit(self.dataset.get_fit_samples())

        inference_components = {
            "preparation_pipeline": preparation_pipeline,
            "preproc_pipeline": preproc_pipeline,
            "model": self.config.model,
        }

        if isinstance(self.model, AbstractClassifier):
            classification_summary: ClassificationSummary = self.model.classification_summary
            model_thresholds = {
                "confidence_threshold": classification_summary.confidence_threshold,
                "eer_threshold": classification_summary.eer_threshold,
                "sensitivity_threshold": classification_summary.sensitivity_threshold,
            }
            inference_components["model_thresholds"] = model_thresholds

        filepath = Path(self.logger.log_dir, "inference_components.pkl")
        with open(filepath, "wb") as file:
            pickle.dump(inference_components, file)

    @property
    def logger(self) -> TensorBoardLogger:
        if self._logger is None:
            self._logger = TensorBoardLogger(save_dir=self.config.log_dir, version=SLURMEnvironment.job_id(),
                                             log_graph=False)
        return self._logger

    def add_model_metadata_for_logger(self, data_loaders: dict[SubsetID, DataLoader]):
        example_datset_id: Optional[SubsetID] = None
        for datset_id in SubsetID:
            if datset_id in data_loaders:
                example_datset_id = datset_id
                break
        example = data_loaders[example_datset_id].dataset[0]

        def replace_tensor_by_noise(_tensor):
            return torch.zeros(self.train_batch_size, *_tensor.shape, dtype=_tensor.dtype)

        if isinstance(example, (tuple, list)) and self.use_multiview:
            example = example[0]

        if isinstance(example, (tuple, list)):
            if self.yield_labels:
                example = example[:-1]
            example = [replace_tensor_by_noise(modality) for modality in example]
            if len(example) == 1:
                example = example[0]
            else:
                example = (example,)
        else:
            example = replace_tensor_by_noise(example)

        self.model.example_input_array = example

    # endregion

    # region Callbacks
    def get_callbacks(self) -> list[pl.callbacks.Callback]:
        callbacks = [
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ]

        if self.config.patience is not None:
            patience_monitor = self.config.log_monitors[0]
            if "validation" not in patience_monitor:
                patience_monitor = "validation_{}_epoch".format(patience_monitor)
            callbacks.append(pl.callbacks.EarlyStopping(monitor=patience_monitor,
                                                        patience=self.config.patience,
                                                        mode=get_monitor_mode(self.config.log_monitors[0])))

        save_last = self.config.save_last or is_representation_model(self.config.model_class_name)
        callbacks += self.get_model_checkpoint_callbacks(save_last=save_last)

        return callbacks

    def get_model_checkpoint_callbacks(self, save_last: bool) -> list[ModelCheckpoint]:
        if self._model_checkpoint_callbacks is None:
            self._model_checkpoint_callbacks = [self.get_model_checkpoint_callback(monitor,
                                                                                   save_last=((i == 0) and save_last))
                                                for i, monitor in enumerate(self.config.log_monitors)]
        return self._model_checkpoint_callbacks

    def get_model_checkpoint_callback(self, monitor: str, save_last: bool):
        checkpoint_kwargs = self.get_model_checkpoint_callback_kwargs(monitor, save_last=save_last)
        if self.config.milestones is None:
            return ModelCheckpoint(**checkpoint_kwargs)
        else:
            return MilestoneCheckpoint(milestones=self.config.milestones, **checkpoint_kwargs)

    def get_model_checkpoint_callback_kwargs(self, monitor: str, save_last: bool):
        if "validation" not in monitor:
            monitor = "validation_{}_epoch".format(monitor)

        checkpoint_kwargs = {
            "dirpath": self.config.checkpoint_dir,
            "filename": "{" + monitor + ":.4f}-{epoch}-{step}",
            "mode": get_monitor_mode(monitor),
            "save_top_k": 1,
            "save_last": save_last,
            "monitor": monitor,
        }
        return checkpoint_kwargs

    # endregion
    # endregion

    # region Config getters
    # region Included tasks
    @property
    def includes_training(self) -> bool:
        return self.config.stages.include_training

    @property
    def includes_testing(self) -> bool:
        return self.config.stages.includes_testing

    @property
    def includes_pseudo_labelling(self) -> bool:
        return self.config.stages.includes_pseudo_labelling

    @property
    def includes_confidence_analysis(self) -> bool:
        return self.config.stages.includes_confidence_analysis

    @property
    def includes_visualization(self) -> bool:
        return self.config.stages.includes_visualization

    @property
    def includes_representation(self) -> bool:
        return self.config.stages.includes_representation

    @property
    def includes_segmentation(self) -> bool:
        return self.config.stages.includes_segmentation

    @property
    def includes_post_training_tasks(self) -> bool:
        return self.includes_training and (
                self.includes_training or
                self.includes_pseudo_labelling or
                self.includes_visualization or
                self.includes_representation or
                self.includes_segmentation
        )

    # endregion

    def get_batch_size(self, datset: MindfulDataset, subset_id: SubsetID):
        if self.config.batch_sizes[subset_id] is not None:
            return self.config.batch_sizes[subset_id]
        else:
            return len(datset.samples[subset_id])

    @property
    def train_batch_size(self) -> int:
        return self.config.batch_sizes[SubsetID.TRAIN]

    @property
    def multiview(self) -> bool | int:
        if self._multiview is None:
            self._multiview = MindfulModule.get_module_class(self.config.model_class_name).multiview()
        return self._multiview

    @property
    def use_multiview(self) -> bool:
        if isinstance(self.multiview, int):
            return self.multiview > 1
        else:
            return self.multiview

    @property
    def yield_labels(self) -> bool:
        return is_classification_model(self.config.model_class_name)

    # endregion
