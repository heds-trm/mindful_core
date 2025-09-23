import argparse
from pathlib import Path
import shutil
from typing import Optional

from data.datasets.protoset import ProtoSet


def prepare_dataset(input_folds_folder: str,
                    pipeline_config_path: str,
                    output_folder: str,
                    extension: str,
                    view_pipeline_config: str = None,
                    view_folder: str = None,
                    preparation_pipeline_export_path: str | None = None,
                    previews_count=4,
                    ) -> list[Path]:
    if preparation_pipeline_export_path is None:
        preparation_pipeline_export_path = output_folder

    proto_dataset = ProtoSet(name="Dataset",
                             original_folds_folder=input_folds_folder,
                             prepared_folds_folder=output_folder,
                             preparation_pipeline_config_path=pipeline_config_path,
                             preparation_pipeline_export_path=preparation_pipeline_export_path,
                             output_extension=extension,
                             preview_pipeline_config=view_pipeline_config,
                             preview_folder=view_folder,
                             preview_count=previews_count)
    folds_paths = proto_dataset.prepare(save_previews=False)
    proto_dataset.save_dataset_preview()
    return folds_paths


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_folds_folder", type=str, required=True)
    arg_parser.add_argument("--pipeline_config", type=str, required=True)
    arg_parser.add_argument("--output_folder", type=str, required=True)
    arg_parser.add_argument("--extension", type=str, default=".mha")
    arg_parser.add_argument("--view_pipeline_config", type=str, default=None)
    arg_parser.add_argument("--view_folder", type=str, default=None)
    arg_parser.add_argument("--previews_count", type=str, default=100)

    args = arg_parser.parse_args()

    input_folds_folder: str = args.input_folds_folder
    pipeline_config_path: str = args.pipeline_config
    output_folder: str = args.output_folder
    extension: str = args.extension
    view_pipeline_config: Optional[str] = args.view_pipeline_config
    view_folder: str | None = args.view_folder
    previews_count = int(args.previews_count)

    if Path(output_folder).exists():
        shutil.rmtree(output_folder)

    if view_folder is not None:
        for filepath in Path(view_folder).glob("*.(png|dcm)"):
            if filepath.is_file():
                filepath.unlink()

    prepare_dataset(input_folds_folder, pipeline_config_path, output_folder, extension,
                    view_pipeline_config, view_folder,
                    preparation_pipeline_export_path=output_folder,
                    previews_count=previews_count)


if __name__ == "__main__":
    main()
