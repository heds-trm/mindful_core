import torch
import numpy as np
from monai.transforms import SaveImage, SpatialPad, Spacing
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

from data import Modality, ModalityType, Sample
from data.transforms.pipeline import Pipeline
from utils.misc import load_json
from utils.imaging import write_slices
from utils.tensor_utils import normalize


def get_pipeline(path: str) -> Pipeline:
    config = load_json(path)
    return Pipeline(config, multiview=False, preprocess_device="cuda")


def run_image_pipeline(pipeline: Pipeline,
                       image_path: dict[Modality, str | Path]
                       ) -> dict[Modality, torch.Tensor]:
    # noinspection PyProtectedMember
    Sample._check_image_path_types(image_path)

    input_modalities = [Modality(ModalityType.IMAGE, modality.id) for modality in pipeline.input_modalities]
    output_modalities = [Modality(ModalityType.IMAGE, modality.id) for modality in pipeline.output_modalities]
    inputs = {modality.id: image_path[modality] for modality in input_modalities}

    outputs = pipeline(inputs)
    if len(output_modalities) == 1:
        outputs = [outputs]

    outputs = dict(zip(output_modalities, outputs))

    return outputs


def get_samples_paths(path: Path, sample_count: int) -> dict[str, Path]:
    if path.is_file():
        if path.suffix == ".csv":
            fold = pd.read_csv(path)
            if "ScanID" in fold.columns:
                fold = fold.set_index("ScanID")

            if "ScanFilepath" in fold:
                samples = fold["ScanFilepath"].to_dict()
            else:
                image_columns = [column for column in fold.columns if column.endswith(":image")]
                if len(image_columns) != 1:
                    raise RuntimeError("Could not identify the image column")
                else:
                    samples = fold[image_columns[0]].to_dict()
        else:
            raise ValueError(path)
        
        samples = {sample_id: Path(filepath) for sample_id, filepath in samples.items()}
    else:
        samples = {filepath.stem: filepath for filepath in path.glob("**/*.*")
                   if filepath.suffix in {".mha", ".mhd",
                                          ".nii", ".nia", ".nii.gz", ".hdr", ".img", ".img.gz",
                                          ".nrrd", ".nhdr",
                                          ".dcm", ".DCM", ".dicom", ".DICOM",
                                          ".hdf", "*.h4", "*.hdf4", "*.h5", "*.hdf5", "*.he4", "*.he5", "*.hd5",
                                          }
                   }
    if len(samples) == 0:
        raise RuntimeError("{} contains no samples".format(path))
    elif len(samples) <= sample_count:
        return samples
    else:
        selected_ids = list(np.random.choice(list(samples.keys()), size=sample_count))
        return {selected_id: samples[selected_id] for selected_id in selected_ids}


def preview_dataset(input_folder: Path,
                    output_folder: Path,
                    pipeline_config_path: str,
                    sample_count: int,
                    save_3d: str = "",
                    save_mean: bool = True,
                    verbose=False,
                    ):
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    pipeline = get_pipeline(pipeline_config_path)
    image_paths = get_samples_paths(input_folder, sample_count)
    sample_ids = list(image_paths.keys())
    image_modality = Modality("image")

    if (save_3d is not None) and (len(save_3d) > 0):
        if not save_3d.startswith("."):
            save_3d = "." + save_3d
        dcm_saver = SaveImage(output_folder,
                              output_postfix="",
                              output_ext=save_3d,
                              resample=False,
                              separate_folder=False,
                              print_log=False)
    else:
        dcm_saver = None

    images: list[torch.Tensor] = []
    for sample_id in tqdm(sample_ids, disable=not verbose, desc="Previewing samples from {}.".format(input_folder)):
        image_path = image_paths[sample_id]
        modalities = run_image_pipeline(pipeline, {image_modality: image_path})
        image = modalities[image_modality]
        write_slices(image, sample_id, output_folder, method="center")

        if dcm_saver is not None:
            output_name = output_folder / sample_id
            dcm_saver(normalize(image) * 255.0, filename=output_name)

        if save_mean:
            images.append(image)

    if save_mean:
        if verbose:
            print("Computing and saving the mean of previews.")
        spacer = Spacing(pixdim=(2.0, 2.0, 2.0))
        images = [spacer(image) for image in images]

        max_size = list(images[0].shape[-3:])
        for image in images:
            image_spatial_size = image.shape[-3:]
            max_size = [max(image_spatial_size[i], max_size[i]) for i in range(3)]

        padder = SpatialPad(spatial_size=max_size)
        images = [padder(image) for image in images]

        mean_image = torch.stack(images, dim=0).mean(dim=0)
        write_slices(mean_image, "mean_preview", output_folder, method="center")

        if dcm_saver is not None:
            dcm_saver(normalize(mean_image) * 255.0, filename=output_folder / ("mean_preview" + save_3d))


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_folder", type=str, nargs="+")
    arg_parser.add_argument("--output_folder", type=str, nargs="+")
    arg_parser.add_argument("--pipeline_config_path", type=str)
    arg_parser.add_argument("--sample_count", type=int, default=1)
    arg_parser.add_argument("--save_3d", type=str, default="")
    arg_parser.add_argument("--save_mean", action="store_true")

    args = arg_parser.parse_args()
    input_folders: list[Path] = [Path(folder) for folder in args.input_folder]
    output_folders: list[Path] = [Path(folder) for folder in args.output_folder]
    pipeline_config_path: str = args.pipeline_config_path
    sample_count: int = int(args.sample_count)
    save_3d: str = args.save_3d
    save_mean: bool = args.save_mean

    for input_folder, output_folder in zip(input_folders, output_folders):
        preview_dataset(input_folder, output_folder, pipeline_config_path, sample_count, save_3d, save_mean,
                        verbose=True)


if __name__ == "__main__":
    main()
