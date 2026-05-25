#!/usr/bin/env python3
"""
Create patient-safe, study-safe, stratified k-fold CSV splits for MURA.

Rules:
- all images from the same study stay in the same partition
- all images from the same patient stay in the same partition
- preserve positive/negative prevalence as much as possible

For each fold i:
- test  = fold i
- validation = fold (i + 1) % k
- train = all remaining folds

Also writes a database-wide CSV:
- mura_body_parts.csv with columns: ScanID,BodyPart

Output CSV columns for folds:
ScanID,SubsetID,Label,image:image
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm


@dataclass
class SampleRow:
    image_path: Path
    scan_id: str
    body_part: str
    label: int
    patient_id: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create stratified patient-safe k-fold CSV splits for MURA."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Path to the root folder of the MURA dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where fold CSV files will be written.",
    )
    parser.add_argument(
        "--k",
        required=True,
        type=int,
        help="Number of folds (must be >= 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the splitter.",
    )
    return parser.parse_args()


def build_scan_id(rel_path: Path) -> str:
    """
    Expected relative path:
    XR_ELBOW/patient00011/study1_negative/image1.png

    ScanID:
    XR_ELBOW_patient00011_study1_negative_image1
    """
    parts = rel_path.parts
    if len(parts) != 4:
        raise ValueError(f"Unexpected file structure: {rel_path}")

    body_part, patient_id, study_id, image_name = parts
    image_stem = Path(image_name).stem
    return f"{body_part}_{patient_id}_{study_id}_{image_stem}"


def infer_label_from_study_dir(study_dir: str) -> int:
    if "_positive" in study_dir:
        return 1
    if "_negative" in study_dir:
        return 0
    raise ValueError(f"Could not infer label from study folder name: {study_dir}")


def discover_samples(data_root: Path) -> List[SampleRow]:
    """
    Returns a flat list of image-level rows.
    Grouping is done later by patient_id.
    Only includes PNG files that are valid and can be successfully read using PIL.
    """
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    image_files = sorted(data_root.rglob("*.png"))
    samples: List[SampleRow] = []

    for img_path in tqdm(image_files, desc="Scanning PNG images"):
        rel = img_path.relative_to(data_root)
        if len(rel.parts) != 4:
            continue

        # Validate PNG file integrity using PIL
        try:
            with Image.open(img_path) as img:
                # Verify it's actually a PNG and can be loaded
                img.verify()
        except (IOError, OSError, Exception) as e:
            print(f"Warning: Invalid or corrupted PNG file {img_path}: {e}")
            continue

        body_part, patient_id, study_dir, _ = rel.parts
        label = infer_label_from_study_dir(study_dir)
        scan_id = build_scan_id(rel)

        samples.append(
            SampleRow(
                image_path=img_path.resolve(),
                scan_id=scan_id,
                body_part=body_part,
                label=label,
                patient_id=patient_id,
            )
        )

    if not samples:
        raise RuntimeError("No valid PNG images were found under the provided data root.")

    return samples


def write_body_parts_csv(samples: List[SampleRow], output_dir: Path):
    """
    Writes a single database-wide CSV:
    ScanID,BodyPart
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "mura_body_parts.csv"

    rows = [
        {"ScanID": sample.scan_id, "BodyPart": sample.body_part}
        for sample in samples
    ]
    rows.sort(key=lambda r: r["ScanID"])

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ScanID", "BodyPart"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path.name}: {len(rows)} rows")


def split_into_folds(samples: List[SampleRow], k: int, seed: int):
    """
    Uses StratifiedGroupKFold so that:
    - all images from one patient stay together
    - class balance is preserved as much as possible
    """
    if k < 3:
        raise ValueError("k must be >= 3 so that train, validation, and test all exist.")

    y = np.array([s.label for s in samples], dtype=int)
    groups = np.array([s.patient_id for s in samples], dtype=str)
    indices = np.arange(len(samples))

    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_indices: List[np.ndarray] = []
    for _, test_idx in splitter.split(indices, y, groups):
        fold_indices.append(np.array(test_idx, dtype=int))

    if len(fold_indices) != k:
        raise RuntimeError(f"Expected {k} folds, got {len(fold_indices)}.")

    return fold_indices


def write_fold_csvs(samples: List[SampleRow], fold_indices: List[np.ndarray], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    k = len(fold_indices)

    # Precompute which base fold each sample belongs to.
    sample_to_base_fold = {}
    for fold_id, idxs in enumerate(fold_indices):
        for idx in idxs:
            sample_to_base_fold[idx] = fold_id

    for test_fold in tqdm(range(k), desc="Writing fold CSV files"):
        valid_fold = (test_fold + 1) % k

        rows = []
        counts = {"train": 0, "validation": 0, "test": 0}
        counts_pos = {"train": 0, "validation": 0, "test": 0}
        counts_neg = {"train": 0, "validation": 0, "test": 0}

        for idx, sample in enumerate(samples):
            base_fold = sample_to_base_fold[idx]

            if base_fold == test_fold:
                subset_id = "test"
            elif base_fold == valid_fold:
                subset_id = "validation"
            else:
                subset_id = "train"

            counts[subset_id] += 1
            if sample.label == 1:
                counts_pos[subset_id] += 1
            else:
                counts_neg[subset_id] += 1

            rows.append(
                {
                    "ScanID": sample.scan_id,
                    "SubsetID": subset_id,
                    "Label": int(sample.label),
                    "image:image": str(sample.image_path),
                }
            )

        rows.sort(key=lambda r: r["ScanID"])

        out_path = output_dir / f"mura_fold_{test_fold:02d}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["ScanID", "SubsetID", "Label", "image:image"],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"{out_path.name}: "
            f"train={counts['train']} (pos={counts_pos['train']}, neg={counts_neg['train']}), "
            f"validation={counts['validation']} (pos={counts_pos['validation']}, neg={counts_neg['validation']}), "
            f"test={counts['test']} (pos={counts_pos['test']}, neg={counts_neg['test']})"
        )


def main():
    args = parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    samples = discover_samples(data_root)
    write_body_parts_csv(samples, output_dir)

    fold_indices = split_into_folds(samples, k=args.k, seed=args.seed)
    write_fold_csvs(samples, fold_indices, output_dir)

    total_pos = sum(s.label for s in samples)
    total_neg = len(samples) - total_pos
    print(f"Done. Wrote {args.k} fold CSV files and mura_body_parts.csv to: {output_dir}")
    print(f"Global totals: items={len(samples)}, pos={total_pos}, neg={total_neg}")


if __name__ == "__main__":
    main()