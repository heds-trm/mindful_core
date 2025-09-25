from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.callbacks import ModelSummary
import argparse

from mindful_core.models.index import get_model, ModuleConfig


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", required=True)
    arg_parser.add_argument("--model_hparams", required=True)
    arg_parser.add_argument("--checkpoint", default=None)
    arg_parser.add_argument("--rep_model", default=None)
    arg_parser.add_argument("--rep_model_hparams", default=None)
    arg_parser.add_argument("--rep_checkpoint", default=None)

    args = arg_parser.parse_args()

    model_class_name: str = args.model
    model_hparams: str = args.model_hparams
    checkpoint: str | None = args.checkpoint
    rep_model_class_name: str | None = args.rep_model
    rep_model_hparams: str | None = args.rep_model_hparams
    rep_checkpoint: str | None = args.rep_checkpoint

    checkpoint, rep_model_class_name, rep_model_hparams, rep_checkpoint = [
        None if ((value is None) or (value == "")) else value
        for value in (checkpoint, rep_model_class_name, rep_model_hparams, rep_checkpoint)
    ]

    model_config: ModuleConfig = {
        "class_name": model_class_name,
        "hparams": model_hparams,
        "checkpoint": checkpoint,
    }

    if (rep_model_class_name is not None) and (rep_model_hparams is not None):
        rep_module_config: ModuleConfig = {
            "class_name": rep_model_class_name,
            "hparams": rep_model_hparams,
            "checkpoint": rep_checkpoint
        }
        model_config["sub_modules"] = {"representation_model": rep_module_config}

    model = get_model(model_config)

    model_summary = summarize(model, max_depth=1)

    # noinspection PyProtectedMember
    summary_data = model_summary._get_summary_data()
    total_parameters = model_summary.total_parameters
    trainable_parameters = model_summary.trainable_parameters
    model_size = model_summary.model_size
    total_training_modes = model_summary.total_training_modes

    ModelSummary.summarize(summary_data,
                           total_parameters,
                           trainable_parameters,
                           model_size,
                           total_training_modes)


if __name__ == "__main__":
    main()
