import SimpleITK
import argparse
import os
import shutil
import pandas as pd
from tqdm import tqdm


def get_total_filecount(folder: str) -> int:
    return sum([len(filenames) for current_folder, directories, filenames in os.walk(folder)])


def convert_image_to_uint8(image: SimpleITK.Image) -> SimpleITK.Image:
    pixel_data = SimpleITK.GetArrayFromImage(image)
    max_intensity = pixel_data.max()
    min_intensity = pixel_data.min()

    image = (image - min_intensity) / (max_intensity - min_intensity) * 255
    image = SimpleITK.Cast(image, SimpleITK.sitkUInt8)

    return image


def convert_mha(source_folder: str, target_folder: str, use_compression: bool, target_format: str) -> None:
    target_extension = "." + target_format
    total_filecount = get_total_filecount(source_folder)
    with tqdm(total=total_filecount, desc="Converting MHA to {}".format(target_format)) as progress_bar:
        for current_folder, directories, filenames in os.walk(source_folder):
            relative_path = os.path.relpath(current_folder, source_folder)
            current_target_folder = os.path.join(target_folder, relative_path)
            current_target_folder = current_target_folder.replace("mha", target_format)
            if not os.path.exists(current_target_folder):
                os.makedirs(current_target_folder)

            for filename in filenames:
                filepath = os.path.join(current_folder, filename)
                if filename.endswith(".mha"):
                    target_filename = filename.replace(".mha", target_extension)
                    target_filepath = os.path.join(current_target_folder, target_filename)

                    progress_bar.set_description("Converting {} to {}.".format(filename, target_filepath))
                    image = SimpleITK.ReadImage(filepath)

                    if target_format == "dcm":
                        image = convert_image_to_uint8(image)

                    SimpleITK.WriteImage(image, target_filepath, useCompression=use_compression)
                elif filename.endswith(".csv"):
                    target_filepath = os.path.join(current_target_folder, filename)
                    progress_bar.set_description("Updating {} and copying to {}.".format(filename, target_filepath))

                    data_frame = pd.read_csv(filepath, header=None)
                    data_frame = data_frame.replace("mha", target_format, regex=True)
                    data_frame.to_csv(target_filepath, index=False, header=False)
                else:
                    target_filepath = os.path.join(current_target_folder, filename)

                    progress_bar.set_description("Copying {} to {}.".format(filename, target_filepath))
                    shutil.copyfile(filepath, target_filepath)
                progress_bar.update(1)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", required=True)
    arg_parser.add_argument("--target", required=True)
    arg_parser.add_argument("--use_compression", default=False, type=bool)
    arg_parser.add_argument("--image_format", default="nrrd")

    args = arg_parser.parse_args()
    source_folder: str = args.source
    target_folder: str = args.target
    use_compression: bool = args.use_compression and args.use_compression == "True"
    image_format: str = args.image_format

    convert_mha(source_folder, target_folder, use_compression, image_format)


if __name__ == "__main__":
    main()
