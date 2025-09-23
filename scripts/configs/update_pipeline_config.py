import argparse
from pathlib import Path
from typing import Union, Optional, Any

from utils.misc import load_json, write_json
from data.transforms.pipeline import PipelineConfig as PipelineConfigV2, StageConfig, TransformConfig

""" V1
{
    "stage_id":
    {
        "modality_id":
        [
            {
                "transform_type": {
                    **kwargs
                }
            }
        ]
    }
}
"""


""" V2
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
                "parameters": {}
            }
        ]
    }
}
"""

TransformParametersV1 = dict[str, Any]
TransformConfigV1 = dict[str, TransformParametersV1]
ComposeConfigV1 = list[TransformConfigV1]
PipelineConfigV1 = dict[str, dict[str, ComposeConfigV1]]


def update_pipeline_config(config: Union[PipelineConfigV1, PipelineConfigV2]) -> Optional[PipelineConfigV2]:
    if not isinstance(config, dict):
        raise ValueError("Expected a dictionary, got a {}.".format(type(config)))

    if ("inputs" in config) and ("stages" in config):
        # Assuming it is already a valid V2 config
        return config

    if all([stage not in config for stage in ["preprocess", "shared_augment", "view_augment"]]):
        return None

    # From now, we assume this is a valid V1 config
    config: PipelineConfigV1
    modalities = {}
    stages: dict[str, StageConfig] = {}

    for stage_id, stage_composes in config.items():
        stages[stage_id] = []
        for modality_id, modality_composes in stage_composes.items():
            if modality_id == "scan":
                modality_id = "image"
                
            if modality_id not in modalities:
                modalities[modality_id] = modality_id

            for transform_config in modality_composes:
                for transform_type, transform_parameters in transform_config.items():
                    updated_transform_config: TransformConfig = {
                        "modalities": [modality_id],
                        "type": transform_type,
                        "parameters": transform_parameters,
                        "outputs": [modality_id],
                    }
                    stages[stage_id].append(updated_transform_config)

    updated_config = {"inputs": modalities,
                      "outputs": modalities,
                      "stages": stages}
    return updated_config


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("path", nargs="+")
    arg_parser.add_argument("--output_folder", default=None)
    args = arg_parser.parse_args()
    
    paths = [Path(path) for path in args.path]
    tmp: list[Path] = []
    for path in paths:
        if path.is_dir():
            for json_path in path.glob("*.json"):
                tmp.append(json_path)
        else:
            tmp.append(path)
    paths = tmp

    output_folder = None if args.output_folder is None else Path(args.output_folder)
    for json_path in paths:
        original_config = load_json(json_path)
        updated_config = update_pipeline_config(original_config)

        if updated_config is None:
            continue

        if output_folder is None:
            output_path = json_path.with_stem(json_path.stem + "_v2")
        else:
            output_path = output_folder / json_path.name

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)

        write_json(output_path, updated_config)


if __name__ == "__main__":
    main()
