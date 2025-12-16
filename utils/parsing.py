import os
import re
import numpy as np
from pathlib import Path
from typing import Sequence


def parse_list(input_str: Sequence[int | str] | str
               ) -> list[int]:
    if isinstance(input_str, str):
        if not input_str.startswith('[') or input_str.endswith(']'):
            raise ValueError("When providing a string, scan_size must match: [dim1, dim2, dim3].")
        input_str = input_str[1:-1]
        input_str = input_str.split(',')

    output_list = [int(x.strip('[').strip(']').strip(',')) if isinstance(x, str) else x for x in input_str]
    return output_list


def parse_last_number(text: str) -> int | None:
    if not isinstance(text, str):
        raise TypeError("Expected `test` to be string, got a {}".format(type(text)))

    decimal_start = None
    found_decimal = False
    for i, character in reversed(list(enumerate(text))):
        if not character.isdecimal():
            decimal_start = i + 1
            break
        else:
            found_decimal = True

    if not found_decimal:
        return None

    if decimal_start is not None:
        text = text[decimal_start:]

    return int(text)


# region Checkpoint path parsing

def parse_checkpoint_path(checkpoint: str | Path | None,
                          monitor: str | list[str] | None = None,
                          use_last: bool = False,
                          verbose: bool = False
                          ) -> str | None:
    """
    If `checkpoint` is a file, returns `checkpoint`. Otherwise, search if the given folder for a checkpoint.
    If `use_last` is True, will return the `last.ckpt` checkpoint. If `use_last` is False, will search for a checkpoint
    corresponding to the given monitor(s). If `monitor` is None, defaults to searching for 1) validation_loss,
    2) validation_auroc and 3) any .ckpt file.

    The search is adapted for PyTorch Lightning folder architecture
    (experiment_folder/lightning_logs/version/checkpoints).
    If multiple versions are available, will default to the last version.

    Returns None if not valid checkpoint can be found.

    :param checkpoint: The path to search the checkpoint in. Should preferably be a version or a checkpoint folder
        directly to avoid picking the wrong version.
    :param monitor: A monitor (or a list of candidates) to select the checkpoint
    :param use_last: If True, returns the checkpoint saved as "last.ckpt"
    :param verbose: If True, will print what checkpoint was found and its associated monitor value if available.
    """
    if checkpoint is None:
        return None

    if isinstance(checkpoint, Path):
        checkpoint = checkpoint.as_posix()

    if checkpoint.endswith(".ckpt"):
        return checkpoint

    folder = os.path.basename(checkpoint)
    if folder == "checkpoints":
        selected_value = None
        if use_last:
            selected_filename = "last.ckpt"
        else:
            if monitor is None:
                monitors = ["validation_loss", "validation_auroc", None]
            else:
                if isinstance(monitor, str):
                    monitors = [monitor]
                else:
                    monitors = list(monitor)

                monitors = ["validation_{}".format(_monitor) if "validation" not in _monitor else _monitor
                            for _monitor in monitors]

            selected_filename, selected_value = get_best_checkpoint(checkpoint, monitors)

        if selected_filename is None:
            filenames = os.listdir(checkpoint)
            filenames = [filename for filename in filenames if filename.endswith(".ckpt")]
            selected_filename = filenames[-1]
            if verbose:
                print("No checkpoint with monitor value found in {}. Defaulting to {}"
                      .format(checkpoint, selected_filename))
        elif verbose:
            if use_last:
                print("Using `{}` as it is the last checkpoint in {}.".format(selected_filename, checkpoint))
            elif selected_value is None:
                print("Using `{}` as the fallback checkpoint in {}.".format(selected_filename, checkpoint))
            else:
                print("Found best checkpoint {} with monitor value of {} in {}."
                      .format(selected_filename, selected_value, checkpoint))

        return os.path.join(checkpoint, selected_filename)

    paths = os.listdir(checkpoint)
    if ("version" in folder) or ("checkpoints" in paths):
        return parse_checkpoint_path(os.path.join(checkpoint, "checkpoints"),
                                     monitor=monitor, use_last=use_last, verbose=verbose)

    if folder == "lightning_logs":
        versions = [path for path in paths if "version" in path]
        prefix_length = len("version_")
        version_numbers = [int(version[prefix_length:]) for version in versions]
        last_version = versions[np.argmax(version_numbers)]
        return parse_checkpoint_path(os.path.join(checkpoint, last_version, "checkpoints"),
                                     monitor=monitor, use_last=use_last, verbose=verbose)

    if "lightning_logs" in paths:
        return parse_checkpoint_path(os.path.join(checkpoint, "lightning_logs"),
                                     monitor=monitor, use_last=use_last, verbose=verbose)

    raise ValueError("Could not parse {} as a checkpoint file/folder.".format(checkpoint))


KNOWN_MONITORS = {
    "loss": "min",
    "eer": "min",
    "accuracy": "max",
    "acc": "max",
    "auroc": "max",
    "auc": "max",
    "roc": "max",
    "specificity": "max",
    "sensitivity": "max"
}


def get_monitor_mode(monitor_name: str) -> str:
    monitor_name = monitor_name.lower()
    for monitor_id, mode in KNOWN_MONITORS.items():
        if monitor_id in monitor_name:
            return mode

    raise ValueError(monitor_name)


def monitor_compare(monitor_name: str, previous: float | None, current: float) -> bool:
    if previous is None:
        return True

    mode = get_monitor_mode(monitor_name)
    if mode == "min":
        return current < previous
    elif mode == "max":
        return current > previous
    else:
        raise NotImplementedError(monitor_name)


def get_best_checkpoint(folder: str | Path,
                        monitors: list[str | None]
                        ) -> tuple[str | None, float | None]:
    """
    Searches in the given folder for a checkpoint for the given monitors.

    If multiple monitors are provided, picks the first monitor found in the folder.

    If multiple checkpoints are found for the given monitor, picks the checkpoint with the "best" monitor value.
    Check `KNOWN_MONITORS` for how the "best" value is defined for each monitor.

    If None is in the list of monitors, it will pick the first file found ending with .ckpt
    (unless one of the previous monitors in the list is found before).

    Returns the best checkpoint path and its associated monitor value, when applicable.

        :param folder: The folder containing the checkpoint(s).
        :param monitors: A list of candidate monitors.
        :returns: A tuple containing the path to the best checkpoint found (or None if none was found)
        and the associated monitor value if applicable.
    """
    filenames = os.listdir(folder)
    filenames = [filename for filename in filenames if filename.endswith(".ckpt")]

    if len(filenames) == 0:
        return None, None

    allow_any = None in monitors
    while None in monitors:
        monitors.remove(None)

    selected_monitor: str | None = None
    for monitor in monitors:
        for filename in filenames:
            if monitor in filename:
                selected_monitor = monitor
                break

        if selected_monitor is not None:
            break

    if selected_monitor is None:
        if allow_any:
            return filenames[0], None
        else:
            return None, None

    filename_pattern = re.compile(r"""(?P<key>\w+)\s*=+\s*'?(?P<value>\d+\.\d+)'?""")
    selected_filename: str | None = None
    selected_value: float | None = None
    for filename in filenames:
        if selected_monitor in filename:
            for pattern_match in filename_pattern.finditer(filename):
                if selected_monitor in pattern_match.group("key"):
                    current_value = float(pattern_match.group("value"))
                    if monitor_compare(selected_monitor, selected_value, current_value):
                        selected_value = current_value
                        selected_filename = filename

    return selected_filename, selected_value


# endregion

def parse_batch_size(batch_size: int | str | None,
                     default_batch_size: int | str | None
                     ) -> int | str:
    if batch_size is not None:
        if batch_size != "full_batch":
            result = int(batch_size)
        else:
            result = batch_size
    elif default_batch_size is not None:
        result = default_batch_size
    else:
        raise ValueError("The batch size must be defined either by --batch_size (default) "
                         "or its specific argument.")
    return result


def safe_int(value: str | None, default=None) -> int | None:
    return int(value) if value else default


def safe_float(value: str | None, default=None) -> float | None:
    return float(value) if value is not None else default


def safe_bool(value: str | None, default: bool) -> bool:
    return value.lower() not in ["false", "0", "no", "disable"] if value else default

def safe_path(value: str | Path | None, default=None) -> Path | None:
    return Path(value) if value is not None else default
