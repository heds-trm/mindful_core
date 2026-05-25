import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy config files and replace keywords."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory where configs will be copied",
    )

    parser.add_argument(
        "--folds_dir",
        type=str,
        required=True,
        help="Replacement for <folds_dir>",
    )

    parser.add_argument(
        "--pipelines_dir",
        type=str,
        required=True,
        help="Replacement for <pipelines_dir>",
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Replacement for <models_dir>",
    )
    
    parser.add_argument(
        "--runs_dir",
        type=str,
        required=True,
        help="Replacement for <runs_dir>",
    )   

    parser.add_argument(
        "--logs_dir",
        type=str,
        required=True,
        help="Replacement for <logs_dir>",
    )       

    parser.add_argument(
        "--config_root",
        type=str,
        default="./configs",
        help="Root config directory (default: ./configs)",
    )

    return parser.parse_args()


def replace_keywords(text, replacements):
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def process_file(src_file: Path, dst_file: Path, replacements: dict):
    try:
        content = src_file.read_text(encoding="utf-8")
        content = replace_keywords(content, replacements)
        dst_file.write_text(content, encoding="utf-8")
    except UnicodeDecodeError:
        # Binary or non-text file → copy as-is
        shutil.copy2(src_file, dst_file)


def copy_and_process(src_root: Path, dst_root: Path, replacements: dict):
    for path in src_root.rglob("*"):
        relative = path.relative_to(src_root)
        target = dst_root / relative

        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            process_file(path, target, replacements)


def main():
    args = parse_args()

    replacements = {
        "<folds_dir>": args.folds_dir,
        "<pipelines_dir>": args.pipelines_dir,
        "<models_dir>": args.models_dir,
        "<runs_dir>": args.runs_dir,  
        "<logs_dir>": args.logs_dir       
    }

    config_root = Path(args.config_root)
    output_root = Path(args.output_dir)

    subdirs = ["runs", "pipelines", "models"]

    for subdir in subdirs:
        src = config_root / subdir
        dst = output_root / subdir

        if not src.exists():
            print(f"[WARNING] {src} does not exist, skipping.")
            continue

        copy_and_process(src, dst, replacements)

    print("Done.")


if __name__ == "__main__":
    main()