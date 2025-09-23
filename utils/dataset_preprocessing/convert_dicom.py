import SimpleITK
from pathlib import Path
import argparse
from tqdm import tqdm


def reorient_3d(image: SimpleITK.Image) -> SimpleITK.Image:
    # removing dimensions beyond 3
    while image.GetDimension() > 3:
        image = image[..., 0]

    identity = SimpleITK.Transform(3, SimpleITK.sitkIdentity)

    extreme_points = [
        image.TransformIndexToPhysicalPoint((0, 0, 0)),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, 0)),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), 0)),
        image.TransformIndexToPhysicalPoint((0, image.GetHeight(), 0)),
        image.TransformIndexToPhysicalPoint((0, 0, image.GetDepth())),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), 0, image.GetDepth())),
        image.TransformIndexToPhysicalPoint((image.GetWidth(), image.GetHeight(), image.GetDepth())),
        image.TransformIndexToPhysicalPoint((0, image.GetHeight(), image.GetDepth()))
    ]

    min_x = min(extreme_points)[0]
    min_y = min(extreme_points, key=lambda p: p[1])[1]
    min_z = min(extreme_points, key=lambda p: p[2])[2]
    max_x = max(extreme_points)[0]
    max_y = max(extreme_points, key=lambda p: p[1])[1]
    max_z = max(extreme_points, key=lambda p: p[2])[2]

    output_size = image.GetSize()
    output_spacing = [(max_x - min_x) / image.GetWidth(), (max_y - min_y) / image.GetHeight(),
                      (max_z - min_z) / image.GetDepth()]

    # Identity cosine matrix
    output_direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    # Minimal x,y coordinates are the new origin.
    output_origin = [min_x, min_y, min_z]

    resampled_image = SimpleITK.Resample(image, output_size, identity,
                                         SimpleITK.sitkLinear,
                                         output_origin, output_spacing,
                                         output_direction)
    return resampled_image


def strip_metadata(image: SimpleITK.Image) -> None:
    for k in image.GetMetaDataKeys():
        image.EraseMetaData(k)


def convert_dicom(source_folder: Path,
                  target_folder: Path,
                  image_format: str = "mha",
                  use_compression: bool = False,
                  apply_reorient: bool = False,
                  keep_relative_path: bool = False,
                  verbose: bool = False,
                  ) -> list[Path]:
    target_extension = "." + image_format

    dicom_filepaths = list(source_folder.rglob("*.dcm"))
    dicom_folders: list[Path] = []
    for dicom_filepath in dicom_filepaths:
        if dicom_filepath.parent not in dicom_folders:
            dicom_folders.append(dicom_filepath.parent)

    results = []
    progress_bar_description = "Converting DICOM to {}".format(image_format.upper())
    for dicom_folder in tqdm(dicom_folders, desc=progress_bar_description, disable=not verbose):
        if keep_relative_path:
            relative_path = dicom_folder.relative_to(source_folder)
            current_target_folder = target_folder / relative_path
        else:
            current_target_folder = target_folder
        current_target_folder.mkdir(parents=True, exist_ok=True)
        target_filename = dicom_folder.stem + target_extension
        target_filepath = current_target_folder / target_filename

        dicom_files = [filepath.as_posix() for filepath in dicom_folder.glob("*.dcm")]
        reader = SimpleITK.ImageSeriesReader()
        reader.SetFileNames(dicom_files)

        image = reader.Execute()
        if apply_reorient:
            image = reorient_3d(image)
            strip_metadata(image)

        SimpleITK.WriteImage(image, target_filepath.as_posix(), use_compression)
        results.append(target_filepath)

    return results


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", required=True)
    arg_parser.add_argument("--target", required=True)
    arg_parser.add_argument("--image_format", default="mha")
    arg_parser.add_argument("--use_compression", action="store_true")
    arg_parser.add_argument("--reorient", action="store_true")
    arg_parser.add_argument("--keep_relative_path", action="store_true")

    args = arg_parser.parse_args()

    source_folder = Path(args.source)
    target_folder = Path(args.target)
    image_format: str = args.image_format
    use_compression: bool = args.use_compression
    apply_reorient: bool = args.reorient
    keep_relative_path: bool = args.keep_relative_path

    convert_dicom(source_folder=source_folder,
                  target_folder=target_folder,
                  image_format=image_format,
                  use_compression=use_compression,
                  apply_reorient=apply_reorient,
                  keep_relative_path=keep_relative_path,
                  verbose=True)


if __name__ == "__main__":
    main()
