import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil

from mindful_core.data import SubsetID, ModalityType
from mindful_core.data.samples import Sample
from mindful_core.data.data_folds import PresetFold
from mindful_core.data.transforms.pipeline import Pipeline, PipelineConfig, ViewOutput, StageID
from mindful_core.experiments.inference import get_image_savers


# TODO: Build a separate class to save modalities.
#   Would call SaveImage (MONAI) to save images.
#   Would save scalar/categorical data to a CSV
#   Would wait for the first sample to build sub-savers (e.g. to know the number of image dims)
#   Would work similarly to VisualizersGroup from visualization

def sample_pipeline(pipeline_config: Pipeline | PipelineConfig,
                    data: list[Sample] | PresetFold | str | Path,
                    target_folder: str | Path,
                    sample_count: int = None,
                    with_replacement: bool = None,
                    multiview: bool | int = False,
                    preprocess_device="cpu",
                    scalar_features_path: str | Path = None,
                    categorical_features_path: str | Path = None,
                    subsets: list[SubsetID | str] = None,
                    verbose: bool = True,
                    clear_target: bool = False,
                    ) -> None:
    # region Initialization
    # region Convert to pipeline if needed
    if isinstance(pipeline_config, Pipeline):
        pipeline = pipeline_config
    else:
        pipeline = Pipeline(pipeline_config, multiview=multiview, preprocess_device=preprocess_device)
    # endregion

    # region Convert to sample list if needed
    if isinstance(data, (str, Path)):
        data = PresetFold(fold_path=data, 
                          scalar_features_path=scalar_features_path, 
                          categorical_features_path=categorical_features_path)
        
    if isinstance(data, PresetFold):
        fit_data = data.samples[SubsetID.TRAIN]
        if subsets is not None:
            subsets = [SubsetID.parse(subset_id) if not isinstance(subset_id, SubsetID) else subset_id
                       for subset_id in subsets]
            data = sum([data.samples[subset_id] for subset_id in subsets], [])
        else:
            data = data.get_all_samples()
    else:
        fit_data = data

    # endregion

    # region Contextual default value for `with_replacement`
    if with_replacement is None:
        with_replacement = (StageID.SHARED_AUGMENT in pipeline.stages) or (StageID.VIEW_AUGMENT in pipeline.stages)

    # endregion

    # region Sampled indices
    if sample_count is None:
        sample_count = len(data)

    indices = np.random.choice(len(data), size=sample_count, replace=with_replacement)

    # endregion

    target_folder = Path(target_folder)
    if target_folder.exists() and clear_target:
        shutil.rmtree(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    image_savers = get_image_savers(pipeline, target_folder, separate_modality_dirs=True)

    pipeline.fit(fit_data)
    # endregion

    variations: dict[str, int] = {sample.id: 0 for sample in data}
    for index in tqdm(indices, disable=not verbose):
        sample: Sample = data[index]
        output = pipeline(sample)

        if pipeline.use_multiview:
            output_views: list[ViewOutput] = list(output)
        else:
            output_views: list[ViewOutput] = [output]
        
        variation = variations[sample.id]
        for i, output_view in enumerate(output_views):
            if isinstance(output_view, torch.Tensor):
                output_view = [output_view]

            for modality_id, modality_value in zip(pipeline.output_modalities, output_view):
                if modality_id.type == ModalityType.IMAGE:
                    saver = image_savers[modality_id]
                    if pipeline.use_multiview:
                        name = "sample_{}-{}_var_{}".format(sample.id, modality_id.id, variation)
                    else:
                        name = "sample_{}-{}_var_{}-{}".format(sample.id, modality_id.id, variation, i)
                    save_path = Path(saver.folder_layout.output_dir, name)
                    saver(modality_value, filename=save_path)
        
        variations[sample.id] += 1


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--pipeline_config")
    arg_parser.add_argument("--fold_path")
    arg_parser.add_argument("--target_folder")

    arg_parser.add_argument("--sample_count", default=None)
    arg_parser.add_argument("--with_replacement", action="store_true")
    arg_parser.add_argument("--multiview", default="False")
    arg_parser.add_argument("--preprocess_device", default="cpu")
    arg_parser.add_argument("--scalar_features_path", default=None)
    arg_parser.add_argument("--categorical_features_path", default=None)
    arg_parser.add_argument("--subsets", nargs="+", default=None)
    arg_parser.add_argument("--silent", action="store_true")
    arg_parser.add_argument("--clear_target", action="store_true")

    args = arg_parser.parse_args()
    pipeline_config = Path(args.pipeline_config)
    fold_path = Path(args.fold_path)
    target_folder = Path(args.target_folder)

    sample_count: int | None = int(args.sample_count) if (args.sample_count is not None) else None
    with_replacement: bool = args.with_replacement
    multiview: bool | int = (int(args.multiview) if args.multiview.isdigit() 
                             else (args.multiview in ("True", "true", "yes", "y")))
    preprocess_device: str = args.preprocess_device
    scalar_features_path: str | None = args.scalar_features_path
    categorical_features_path: str | None = args.categorical_features_path
    subsets: list[str] | None = args.subsets if ((args.subsets is not None) and (len(args.subsets) > 0)) else None
    verbose: bool = not args.silent
    clear_target: bool = args.clear_target

    sample_pipeline(pipeline_config, fold_path, target_folder,
                    sample_count, with_replacement, multiview, preprocess_device,
                    scalar_features_path, categorical_features_path,
                    subsets, verbose, clear_target)

if __name__ == "__main__":
    main()
