from monai.transforms import SaveImage
import pickle
from tqdm import tqdm
from pathlib import Path
import shutil
from typing import Any

from mindful_core.data import Modality, ModalityType
from mindful_core.data.samples import Sample
from mindful_core.data.data_folds import PresetFold
from mindful_core.data.transforms.pipeline import Pipeline
from mindful_core.data.datasets.preview_dataset import get_pipeline, run_image_pipeline, preview_dataset


# Note : "prepared" here means the dataset has gone through a (pre-)preprocessing pipeline.
# Intended for flipping dataset, pre-cropping, ...
class ProtoSet(object):
    def __init__(self,
                 name: str,
                 original_folds_folder: str | Path,
                 prepared_folds_folder: str | Path = None,
                 preparation_pipeline_config_path: str = None,
                 preparation_pipeline_export_path: str = None,
                 output_extension=".mha",
                 preview_pipeline_config: str = None,
                 preview_folder: str | Path = None,
                 preview_count: int = None,
                 scalar_features_path: str = None,
                 categorical_features_path: str = None):
        self.name = name

        self.original_folds_folder = Path(original_folds_folder)
        self.prepared_folds_folder = Path(prepared_folds_folder) if (prepared_folds_folder is not None) else None
        self.prepared_folds_paths: list[Path] | None = None
        self.preparation_pipeline_config_path = preparation_pipeline_config_path

        if preparation_pipeline_export_path is not None:
            if preparation_pipeline_export_path != "":
                export_path = Path(preparation_pipeline_export_path)
                if len(export_path.suffix) == 0:
                    export_path = export_path / "preparation_pipeline.pkl"
                self.preparation_pipeline_export_path = export_path.as_posix()
            else:
                self.preparation_pipeline_export_path = None
        else:
            if self.preparation_pipeline_config_path is not None:
                raise ValueError("You must define a path to where the preparation pipeline will be exported.")
            self.preparation_pipeline_export_path = None

        self._preparation_pipeline: Pipeline | None = None
        self.output_extension = output_extension

        self.preview_pipeline_config = preview_pipeline_config
        if preview_folder is None:
            if prepared_folds_folder is not None:
                preview_folder = self.prepared_folds_folder / "previews"
            else:
                preview_folder = None
        else:
            preview_folder = Path(preview_folder)
        self.preview_folder = preview_folder
        self.preview_count = preview_count or 10

        self.scalar_features_path = scalar_features_path
        self.categorical_features_path = categorical_features_path

    def get_experiment_folds(self,
                             folds_indices: int | str | list[int],
                             verbose=True
                             ) -> dict[int, str]:
        folds_paths = [path.as_posix() for path in self.prepare(verbose)]

        if isinstance(folds_indices, str):
            if folds_indices.isnumeric():
                folds_indices = int(folds_indices)
            elif folds_indices == "all":
                folds_indices = list(range(len(folds_paths)))
            else:
                raise ValueError("Unexpected value for `step_folds`: {}".format(folds_indices))

        if isinstance(folds_indices, int):
            return {folds_indices: folds_paths[folds_indices]}

        if isinstance(folds_indices, (list, tuple)):
            return {i: folds_paths[i] for i in folds_indices}

        raise TypeError("Unexpected type for `step_folds`: {}".format(type(folds_indices)))

    def prepare(self, verbose=True, save_previews=True) -> list[Path]:
        if self.check_prepared():
            if verbose:
                print("{} - {}, skipping preparation.".format(self.name, self.get_reason_why_already_prepared()))
            return self.prepared_folds_paths or self.list_folds(self.original_folds_folder)

        folds_paths, folds = self.get_original_folds()

        self.prepare_destination_folders()

        image_filepath_map = self.prepare_samples(folds, verbose=verbose)
        self.prepared_folds_paths = self.remap_and_save_prepared_folds(folds_paths, folds, image_filepath_map)

        if save_previews:
            self.save_dataset_preview(verbose=verbose)

        return self.prepared_folds_paths

    # region Preparation methods
    # region Check if not already prepared
    def check_prepared(self) -> bool:
        if self.no_preparation_needed or self.already_prepared:
            return True

        if not self.prepared_scans_folder_exists:
            return False

        existing_folds_paths, existing_folds = self.get_prepared_folds()
        original_folds_paths, original_folds = self.get_original_folds()

        # region Check for eventual fold/samples mismatches
        if len(existing_folds_paths) != len(original_folds_paths):
            return False

        for existing_fold, original_fold in zip(existing_folds_paths, original_folds_paths):
            if existing_fold.name != original_fold.name:
                return False

        existing_samples = self.get_unique_samples(existing_folds)
        for existing_sample in existing_samples:
            for _, image_path in existing_sample.image_path.items():
                if not Path(image_path).exists():
                    return False
        # endregion

        # region Restore missing links when possible
        original_samples = self.get_unique_samples(original_folds)
        existing_samples_ids = [sample.id for sample in existing_samples]
        original_samples_ids = [sample.id for sample in original_samples]

        if (all([scan_id in existing_samples_ids for scan_id in original_samples_ids]) and
                all([scan_id in original_samples_ids for scan_id in existing_samples_ids])):
            self.prepared_folds_paths = existing_folds_paths
        # endregion

        return True

    @property
    def no_preparation_needed(self) -> bool:
        return (self.prepared_folds_folder is None) or \
            (self.preparation_pipeline_config_path is None)

    @property
    def preparation_pipeline(self) -> Pipeline | None:
        return self._preparation_pipeline

    @property
    def already_prepared(self) -> bool:
        return self.prepared_folds_paths is not None

    @property
    def prepared_scans_folder_exists(self) -> bool:
        return self.prepared_folds_folder.exists()

    def get_reason_why_already_prepared(self) -> str:
        if self.prepared_folds_folder is None:
            reason = "No preparation folder set"
        elif self.preparation_pipeline_config_path is None:
            reason = "No preparation pipeline set"
        else:
            reason = "Located prepared dataset in {}".format(self.prepared_folds_folder)
        return reason

    # endregion

    def prepare_destination_folders(self) -> None:
        if self.prepared_folds_folder.exists():
            shutil.rmtree(self.prepared_folds_folder)

    def prepare_samples(self, folds: list[PresetFold], verbose=True) -> dict[str, dict[Modality, Path]]:
        if len(folds) == 0:
            raise ValueError("No folds were provided.")

        original_samples = self.get_unique_samples(folds)

        pipeline = self._make_preparation_pipeline(original_samples)

        image_modalities = [modality for modality in pipeline.output_modalities if modality.type == ModalityType.IMAGE]
        savers: dict[Modality, SaveImage] = self.get_image_savers(image_modalities)

        image_filepath_map: dict[str, dict[Modality, Path]] = {}
        description = "{} - Preparing samples".format(self.name)
        for original_sample in tqdm(original_samples, desc=description, disable=not verbose):
            output_images = run_image_pipeline(pipeline, original_sample.image_path)

            sample_output_paths: dict[Modality, Path] = {}
            for modality in output_images:
                output_image = output_images[modality]
                output_folder = self.get_prepared_images_folder(modality)
                output_filepath = output_folder / original_sample.id

                savers[modality](output_image, filename=output_filepath)

                output_filepath = output_folder / (original_sample.id + self.output_extension)
                sample_output_paths[modality] = output_filepath
            image_filepath_map[original_sample.id] = sample_output_paths

        return image_filepath_map

    def _make_preparation_pipeline(self, samples: list[Sample]) -> Pipeline:
        if (self._preparation_pipeline is not None) and (self._preparation_pipeline.is_fit()):
            raise RuntimeError("Tried to prepare a dataset using the same pipeline more than once.")

        pipeline = get_pipeline(self.preparation_pipeline_config_path)
        pipeline.fit(samples)

        if self.preparation_pipeline_export_path is not None:
            export_folder = Path(self.preparation_pipeline_export_path).parent
            export_folder.mkdir(parents=True, exist_ok=True)
            with open(self.preparation_pipeline_export_path, "wb") as file:
                # noinspection PyTypeChecker
                pickle.dump(pipeline, file)

        self._preparation_pipeline = pipeline
        return pipeline

    def remap_and_save_prepared_folds(self,
                                      original_folds_paths: list[Path],
                                      original_folds: list[PresetFold],
                                      image_filepath_map: dict[str, dict[Modality, Path]]) -> list[Path]:
        prepared_folds_paths = []
        for fold_path, fold in zip(original_folds_paths, original_folds):
            for subset_id in fold.samples:
                for sample in fold.samples[subset_id]:
                    sample.image_path = image_filepath_map[sample.id]

            prepared_fold_path: Path = self.prepared_folds_folder / fold_path.name
            fold.save_scan_data(prepared_fold_path.as_posix())
            prepared_folds_paths.append(prepared_fold_path)

        return prepared_folds_paths

    def save_dataset_preview(self, verbose=True):
        if (self.preview_pipeline_config is not None) and (self.preview_folder is not None):
            if verbose:
                print("Saving previews of `{}` in {}.".format(self.name, self.preview_folder))
            pipeline = get_pipeline(self.preparation_pipeline_config_path)
            for modality in pipeline.input_modalities:
                input_folder = self.get_prepared_images_folder(modality)
                preview_dataset(input_folder, self.preview_folder,
                                self.preview_pipeline_config, sample_count=self.preview_count)

    # endregion

    # region Get folds/samples
    def get_original_folds(self) -> tuple[list[Path], list[PresetFold]]:
        return self.get_folds(self.original_folds_folder)

    def get_prepared_folds(self) -> tuple[list[Path], list[PresetFold]]:
        return self.get_folds(self.prepared_folds_folder)

    # region Static helper methods
    @staticmethod
    def get_folds(folds_folder: Path) -> tuple[list[Path], list[PresetFold]]:
        folds_paths = ProtoSet.list_folds(folds_folder)
        folds = ProtoSet.load_folds(folds_paths)
        return folds_paths, folds

    @staticmethod
    def load_folds(folds_paths: list[Path]) -> list[PresetFold]:
        return [PresetFold(input_fold_path.as_posix()) for input_fold_path in sorted(folds_paths)]

    @staticmethod
    def list_folds(folds_folder: Path) -> list[Path]:
        folds_paths = []
        for filepath in folds_folder.iterdir():
            if filepath.match("*fold_*.csv"):
                folds_paths.append(filepath)
        return list(sorted(folds_paths))

    @staticmethod
    def get_unique_samples(folds: list[PresetFold]) -> list[Sample]:
        if len(folds) == 0:
            raise ValueError("No folds were provided.")

        unique_samples = folds[0].get_all_samples()
        sample_ids = [sample.id for sample in unique_samples]

        for input_fold in folds[1:]:
            fold_samples = input_fold.get_all_samples()
            for sample in fold_samples:
                if sample.id not in sample_ids:
                    sample_ids.append(sample.id)
                    unique_samples.append(sample)

        return unique_samples

    # endregion
    # endregion

    # region Scan saver/destination folder
    def get_image_savers(self, image_modalities: list[Modality]) -> dict[Modality, SaveImage]:
        return {
            modality:
                SaveImage(self.get_prepared_images_folder(modality).as_posix(),
                          output_ext=self.output_extension,
                          output_postfix="",
                          resample=False,
                          separate_folder=False,
                          print_log=False)
            for modality in image_modalities
        }

    def get_prepared_images_folder(self, modality: Modality | str) -> Path:
        if isinstance(modality, Modality):
            modality = modality.id
        folder: Path = self.prepared_folds_folder / modality
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    # endregion

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "ProtoSet":
        return ProtoSet(name=name,
                        original_folds_folder=config["folds"],
                        prepared_folds_folder=config.get("prepared_folds"),
                        preparation_pipeline_config_path=config.get("preparation_pipeline"),
                        preparation_pipeline_export_path=config.get("preparation_pipeline_export"),
                        output_extension=config.get("extension", ".mha"),
                        preview_pipeline_config=config.get("preview_pipeline"),
                        preview_folder=config.get("preview_folder"),
                        preview_count=config.get("preview_count"),
                        scalar_features_path=config.get("scalar_features"),
                        categorical_features_path=config.get("categorical_features"))


class ProtoMetaSet(ProtoSet):
    def __init__(self,
                 proto_sets: list[ProtoSet],
                 original_folds_folder: str | Path,
                 prepared_folds_folder: str | Path = None,
                 scalar_features_path: str = None,
                 categorical_features_path: str = None,
                 ):
        self.proto_sets = proto_sets
        super(ProtoMetaSet, self).__init__(name="+".join([proto_set.name for proto_set in self.proto_sets]),
                                           original_folds_folder=original_folds_folder,
                                           prepared_folds_folder=prepared_folds_folder,
                                           scalar_features_path=scalar_features_path,
                                           categorical_features_path=categorical_features_path)

    def prepare_samples(self, folds: list[PresetFold], verbose=True) -> dict[str, dict[Modality, str]]:
        individual_folds = {proto_set.name: self.load_folds(proto_set.prepare(verbose))
                            for proto_set in self.proto_sets}

        scan_filepath_map: dict[str, dict[Modality, str]] = {}
        for dataset_name, dataset_folds in individual_folds.items():
            dataset_unique_samples = self.get_unique_samples(dataset_folds)
            for sample in dataset_unique_samples:
                scan_id = "{}_{}".format(dataset_name, sample.id)
                scan_filepath_map[scan_id] = sample.image_path

        return scan_filepath_map

    @property
    def no_preparation_needed(self) -> bool:
        return self.prepared_folds_folder is None

    def get_reason_why_already_prepared(self) -> str:
        if self.prepared_folds_folder is None:
            reason = "No preparation folder set"
        else:
            reason = "Located prepared dataset in {}".format(self.prepared_folds_folder)
        return reason

    @staticmethod
    def from_proto_sets_and_config(proto_sets: list[ProtoSet], config: dict[str, Any]) -> "ProtoMetaSet":
        return ProtoMetaSet(proto_sets=proto_sets,
                            original_folds_folder=config["folds"],
                            prepared_folds_folder=config.get("prepared_folds"),
                            scalar_features_path=config.get("scalar_features"),
                            categorical_features_path=config.get("categorical_features"))

    def prepare_destination_folders(self) -> None:
        if self.prepared_folds_folder.exists():
            shutil.rmtree(self.prepared_folds_folder)
        self.prepared_folds_folder.mkdir(parents=True)


def get_proto_datasets(datasets_config: dict[str, dict[str, str]]) -> dict[str, ProtoSet]:
    individual_proto_datasets = {dataset_name: ProtoSet.from_config(dataset_name, dataset_config)
                                 for dataset_name, dataset_config in datasets_config.items()
                                 if "+" not in dataset_name}
    proto_meta_datasets = {}
    for dataset_name, dataset_config in datasets_config.items():
        if ("+" not in dataset_name) or dataset_name.startswith("-"):
            continue

        subsets_names = dataset_name.split("+")
        missing_subsets = [subset_name for subset_name in subsets_names if subset_name not in individual_proto_datasets]
        if len(missing_subsets) > 0:
            raise RuntimeError("You are missing the following base datasets: {}.".format(", ".join(missing_subsets)))

        needed_proto_sets = [individual_proto_datasets[subset_name] for subset_name in subsets_names]
        proto_meta_datasets[dataset_name] = ProtoMetaSet.from_proto_sets_and_config(needed_proto_sets, dataset_config)

    proto_datasets = {**individual_proto_datasets, **proto_meta_datasets}
    return proto_datasets
