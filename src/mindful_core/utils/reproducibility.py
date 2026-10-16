from pathlib import Path
import shutil
import tempfile
from types import ModuleType


def get_ignored_patterns(git_ignore_path: Path | None, ignore_hidden: bool) -> list[str]:
    if ignore_hidden:
        ignored_patterns = [".*"]
    else:
        ignored_patterns = [".git", ".gitignore"]

    if (git_ignore_path is None) or (not git_ignore_path.exists()):
        return ignored_patterns

    with open(git_ignore_path, "r") as git_ignore_file:
        git_ignore_lines = git_ignore_file.readlines()

    for line in git_ignore_lines:
        if line.startswith("#"):
            continue

        line = line.replace("\n", "").replace("\r", "").replace("\t", "")
        if len(line) == 0:
            continue

        line = line.strip()
        if line not in ignored_patterns:
            ignored_patterns.append(line)

    return ignored_patterns


def path_matches_any_pattern(path: Path, patterns: list[str]) -> bool:
    for pattern in patterns:
        if path.match(pattern):
            return True
    return False


def list_filepaths(root_dir: Path, ignored_patterns: list[str]) -> list[Path]:
    remaining_directories = [Path(root_dir)]
    filepaths = []

    while len(remaining_directories) > 0:
        current_directory = remaining_directories.pop(0)

        for path in current_directory.glob("*"):
            if path_matches_any_pattern(path, ignored_patterns):
                continue

            if path.is_dir():
                remaining_directories.append(path)
            else:
                filepaths.append(path)

    return filepaths


def archive_modules(modules: list[ModuleType], destination: str | Path) -> None:
    # region Parse destination
    destination = Path(destination)
    if len(destination.suffix) > 0:
        archive_format = destination.suffix[1:]
        destination = destination.with_suffix("")
    else:
        archive_format = "zip"
        if destination.exists():
            destination = destination / "code"
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)

    # endregion

    with tempfile.TemporaryDirectory() as tmp_folder:
        for module in modules:
            origin = Path(module.__file__).parent
            git_ignore_path = origin / ".gitignore"
            ignored_patterns = get_ignored_patterns(git_ignore_path, ignore_hidden=True)
            new_module_root = Path(tmp_folder, module.__name__)

            filepaths = list_filepaths(origin, ignored_patterns)
            for filepath in filepaths:
                if filepath.suffix != ".py":
                    continue

                new_path = Path(new_module_root, filepath.relative_to(origin))
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(filepath, new_path)

        shutil.make_archive(destination.as_posix(), archive_format, tmp_folder)
