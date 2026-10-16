import torch
import argparse

from mindful_core.experiments import ExperimentSeries
from mindful_core.utils.misc import initialize_tensorboard


def main():
    initialize_tensorboard()
    torch.multiprocessing.set_sharing_strategy("file_system")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config", help="The path(s) to experiment series "
                                           "configuration file(s) (JSON format).", nargs="+")

    args = arg_parser.parse_args()
    configs: list[str] = args.config

    for config in configs:
        experiment_series = ExperimentSeries(config)
        experiment_series.run()


if __name__ == "__main__":
    main()

# TODO : Add documentation for all classes and utility functions

# TODO : Improve dataset collections (notably automate the fusion, giving the ability to set a dataset as the test set)
# TODO : Improve how scalar / categorical data is linked to dataset ?

# TODO : Experiment round stages - Ability to config stages (use a dict)

# TODO : Shared experiment parameters - Define a base to re-use
#  (should work across shared config also to be able to define a default config)

