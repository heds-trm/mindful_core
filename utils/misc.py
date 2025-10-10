import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
import json
from json.decoder import JSONDecodeError
import sys
from contextlib import contextmanager
import importlib.util
from abc import ABCMeta
from typing import TypeVar, Any, Type

_T = TypeVar("_T")


def find_all_subclasses(cls: type[_T]) -> set[type[_T]]:
    return set(cls.__subclasses__()).union([sub_sub_class for sub_class in cls.__subclasses__()
                                            for sub_sub_class in find_all_subclasses(sub_class)])


def get_abstract_methods(obj: Type[Type[ABCMeta]]) -> list[str]:
    if not isinstance(obj, type):
        raise ValueError("`obj` is not a type, got {}".format(type(obj)))

    if not issubclass(type(obj), ABCMeta):
        raise ValueError("`obj` must derive from ABC.")

    abstract_methods = []
    for name, value in obj.__dict__.items():
        if getattr(value, "__isabstractmethod__", False):
            abstract_methods.append(value)

    for base in obj.__bases__:
        for name in getattr(base, "__abstractmethods__", ()):
            value = getattr(obj, name, None)
            if value in abstract_methods:
                continue

            if getattr(value, "__isabstractmethod__", False):
                abstract_methods.append(value)

    abstract_methods = [method.__qualname__ for method in abstract_methods]
    return abstract_methods


# region Filepaths
def find_first_path(folder: str | Path, pattern: str, recursive: bool = False) -> Path | None:
    folder = Path(folder)

    iterator = folder.rglob(pattern=pattern) if recursive else folder.glob(pattern=pattern)
    for filepath in iterator:
        return filepath

    return None


def get_filepath_suffix(filepath: str | Path):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    return filepath.suffix


# endregion

# region JSON
def load_json(path: str | Path) -> dict[Any, Any]:
    try:
        with open(path, "r") as file:
            return json.load(file)
        
    except JSONDecodeError as exception:
        raise JSONDecodeError("Could not decode the following JSON file: {}".format(path)) from exception
    
    except PermissionError as exception:
        if Path(path).is_dir():
            raise IsADirectoryError("`{}` is a directory, expected a file.".format(path))
        else:
            raise exception
    

def try_load_json(path: str | Path, file_description: str) -> dict[Any, Any]:
    try:
        with open(path, "r") as file:
            return json.load(file)
        
    except JSONDecodeError as exception:
        error_message = "Could not decode `{}` file at `{}`".format(file_description, path)
        raise JSONDecodeError(error_message) from exception
    
    except FileNotFoundError as exception:
        error_message = "No such `{}` file: `{}`".format(file_description, path)
        raise FileNotFoundError(error_message) from exception

    except PermissionError as exception:
        if Path(path).is_dir():
            raise IsADirectoryError("Path to `{}` ({}) is a directory, expected a file.".format(file_description, path))
        else:
            raise exception


def write_json(path: str | Path, value: dict[Any, Any]) -> None:
    with open(path, 'w') as file:
        json.dump(value, file, indent=4)


# endregion

# region Tables (Pandas)
TablePath = list[str | Path] | str | Path


def load_table(table_path: TablePath | None,
               index_col: str | None = None,
               sheet_name: str | int = 0) -> pd.DataFrame | None:
    if table_path is None:
        return None

    if isinstance(table_path, (list, tuple)):
        tables = [load_table(filepath, index_col=index_col, sheet_name=sheet_name) for filepath in table_path]
        tables = [table for table in tables if table is not None]
        columns = list(tables[0].columns)
        for table in tables[1:]:
            other_columns = list(table.columns)
            for column in other_columns:
                if column not in columns:
                    raise RuntimeError("Could not find column `{}` in all tables.".format(column))
            for column in columns:
                if column not in other_columns:
                    raise RuntimeError("Could not find column `{}` in all tables.".format(column))
        table = pd.concat(tables)
        return table

    suffix = get_filepath_suffix(table_path)
    if suffix == ".csv":
        table = pd.read_csv(table_path, index_col=index_col)
    elif suffix == ".xlsx":
        table = pd.read_excel(table_path, index_col=index_col, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported extension `{}` for filepath `{}`".
                         format(suffix, table_path))

    return table


def save_csv(path: Path | str,
             data_frame: pd.DataFrame,
             fallback_name: str = "data_frame",
             save_index=False):
    output_path = Path(path)

    if output_path.suffix == ".csv":
        output_folder = output_path.parent
        output_file = output_path
    elif output_path.suffix == "":
        output_folder = output_path
        fallback_name = fallback_name if fallback_name.endswith(".csv") else "{}.csv".format(fallback_name)
        output_file = output_path / fallback_name
    else:
        output_folder = output_path.parent
        output_file = output_path.with_suffix(".csv")

    output_folder.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_file, index=save_index)


# endregion

# region Set value for context (with ContextValue(...) :)

class DynValue(object):
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return self.value

    def __contains__(self, value: Any):
        return value in self.value


class ContextValue(object):
    def __init__(self, ref: DynValue, context_value: Any):
        self.ref = ref
        self.previous = ref.value
        self.context_value = context_value

    def __enter__(self):
        self.ref.value = self.context_value

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ref.value = self.previous


# endregion

# region Python package checks

def check_packages(names: str | list[str]) -> list[bool]:
    if isinstance(names, str):
        names = [names]

    return [check_package(name) for name in names]


def check_package(name: str) -> bool:
    if name in sys.modules:
        return True

    return importlib.util.find_spec(name) is not None


# endregion

# region Tensorboard (Tensorflow / Torch)

def initialize_tensorboard(verbose: bool = False):
    try:
        # noinspection PyUnresolvedReferences, PyPackageRequirements
        import tensorflow as tf  # type:ignore
        import tensorboard as tb
        # noinspection PyUnresolvedReferences
        tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # type: ignore
        if verbose:
            print("Tensorflow found, using Tensorflow's base Tensorboard.")
    except ModuleNotFoundError:
        if verbose:
            print("Tensorflow not found, using PyTorch's base Tensorboard.")


# endregion


def is_defined(record_value: Any) -> bool:
    if isinstance(record_value, list):
        return any([value is not None for value in record_value])
    return (not isinstance(record_value, float)) or (not np.isnan(record_value))


def generate_binary_combinations(length: int) -> list[list[bool]]:
    return [[bool(i & 2 ** j) for j in reversed(range(length))] for i in range(2 ** length)]
