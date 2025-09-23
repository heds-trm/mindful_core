import pandas as pd
# noinspection PyPackageRequirements
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
from pathlib import Path


def gather_representations(path: Path):
    log_path = path / "lightning_logs"

    representation_paths = [version_path / "representations.csv" for version_path in log_path.iterdir()
                            if (version_path / "representations.csv").exists()]
    representations = [pd.read_csv(representation_path, index_col="ScanID")
                       for representation_path in representation_paths]
    
    for i, representation_set in enumerate(representations):
        representation_set = representation_set.drop(columns="SubsetID")
        representation_set.to_csv(path / "representations_{:02d}.csv".format(i))

        scan_ids = representation_set.index
        representation_set = np.asarray(representation_set)
        representation_set = StandardScaler().fit_transform(representation_set)

        reducer = umap.UMAP(n_components=8)
        reduced = reducer.fit_transform(representation_set)

        reduced = pd.DataFrame(data=reduced, index=scan_ids)
        reduced.to_csv(path / "reduced_representations_{:02d}.csv".format(i))


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", required=True, nargs="+")
    args = arg_parser.parse_args()

    paths = args.path
    for path in paths:
        gather_representations(Path(path))


if __name__ == "__main__":
    main()
