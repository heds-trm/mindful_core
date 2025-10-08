from pathlib import Path
import argparse
import copy
import warnings

from mindful_core.utils.misc import try_load_json, write_json


def make_follow_up_experiment_series_config(config_path: Path,
                                            mode: str, 
                                            dataset_filter: list[str],
                                            mode_filter: list[str]
                                            ) -> Path:
    warnings.warn(("`make_follow_up_experiment_series_config` is currently deprecated, "
                   "as it relies on a deprecated schema."),
                  category=DeprecationWarning)

    mode = mode.replace(" ", "-")
    run_config = try_load_json(config_path, "Experiment Series config")

    steps_config: dict[str, dict] = run_config["steps"]
    steps = list(steps_config.keys())
    for step in steps:
        step_config = steps_config[step]

        if len(dataset_filter) > 0:
            if step_config["dataset"] not in dataset_filter:
                steps_config.pop(step)
                continue

        if len(mode_filter) > 0:
            step_mode = step_config["mode"] if "mode" in step_config else run_config["shared"]["mode"]
            if not all([_filter in step_mode for _filter in mode_filter]):
                continue

        step_config["skip"] = "yes"
        
        follow_up_config = copy.deepcopy(step_config)
        follow_up_config["skip"] = "no"
        follow_up_config["mode"] = mode
        follow_up_config["checkpoint"] = "${}".format(step)

        if "rep_checkpoint" in follow_up_config:
            follow_up_config.pop("rep_checkpoint")

        steps_config["{}-{}".format(step, mode)] = follow_up_config

    if len(dataset_filter) > 0:
        datasets: dict = run_config["datasets"]
        for dataset in list(datasets.keys()):
            if dataset not in dataset_filter:
                datasets.pop(dataset)

    output_path = config_path.parent / "{}_follow_up.json".format(config_path.stem)
    write_json(output_path, run_config)

    return output_path


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--base_run_config", required=True, type=str)
    arg_parser.add_argument("--mode", required=True, type=str)
    arg_parser.add_argument("--dataset_filter", default=[], nargs="+")
    arg_parser.add_argument("--mode_filter", default=[], nargs="+")
    args = arg_parser.parse_args()

    config_path = Path(args.base_run_config)
    mode: str = args.mode
    dataset_filter: list[str] = args.dataset_filter
    mode_filter: list[str] = args.mode_filter

    output_path = make_follow_up_experiment_series_config(config_path, mode, dataset_filter, mode_filter)
    print(output_path)
    

if __name__ == "__main__":
    main()
