from pathlib import Path
import argparse

from mindful_core.utils.dicom import compare_dicom_fields


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("root")
    args = arg_parser.parse_args()
    root = Path(args.root)
    to_compare = list(Path(root).rglob(pattern="*.dcm"))
    result = compare_dicom_fields(to_compare)
    print(result)


if __name__ == "__main__":
    main()
