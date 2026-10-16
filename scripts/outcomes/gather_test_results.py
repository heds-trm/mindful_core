import argparse

from mindful_core.utils.reporting import gather_test_results


def main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--root", required=True, type=str)
    arg_parser.add_argument("--summarize_folds", default=True, type=bool)
    arg_parser.add_argument("--save_absolute_path", default=False)
    arg_parser.add_argument("--save_folder", default=None, type=str)
    arg_parser.add_argument("--min_folds", default=-1, type=int)
    arg_parser.add_argument("--datasets", nargs="+")

    args = arg_parser.parse_args()

    root: str = args.root
    summarize_folds: bool = args.summarize_folds
    save_absolute_path: bool = args.save_absolute_path in ["True", "true", 1, "Yes", "yes"]
    save_folder: str | None = args.save_folder
    datasets: list[str] = args.datasets
    min_folds: int = args.min_folds or -1

    gather_test_results(root, summarize_folds, save_absolute_path, save_folder, datasets, min_folds)


if __name__ == "__main__":
    main()
