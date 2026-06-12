"""Filter the full MIMIC-CXR CheXpert label sheet down to just the
study_ids in our pretraining subset, so a tiny labels file can be
committed and shipped to Colab via `git clone` (no PhysioNet
credentials or Drive upload needed at train time).

Run this ONCE locally after you have the credentialed PhysioNet file:

    mimic-cxr-2.0.0-chexpert.csv      (MIMIC-CXR-JPG v2.0.0 on PhysioNet)

Usage:
    python biovilt/make_chexpert_subset.py \
        --chexpert /path/to/mimic-cxr-2.0.0-chexpert.csv \
        --subset-dir subset_out \
        --out subset_out/chexpert_labels_subset.csv

The output keeps the original CheXpert columns (subject_id, study_id +
14 disease columns) and only the rows whose study_id appears in any of
the train/val/test image-level CSVs.

NOTE: The full MIMIC CheXpert CSV is credentialed data — do NOT commit
it. The *filtered* subset file contains only labels for studies already
present in your committed subset CSVs, so it carries no new exposure.
"""
import argparse
import glob
import os
import sys

import pandas as pd


def collect_subset_study_ids(subset_dir: str) -> set:
    """Union of study_id over every *_imagelevel.csv in subset_dir."""
    paths = sorted(glob.glob(os.path.join(subset_dir, "*_imagelevel.csv")))
    if not paths:
        sys.exit(f"No *_imagelevel.csv files found in {subset_dir!r}.")
    ids: set = set()
    for p in paths:
        df = pd.read_csv(p, usecols=["study_id"])
        ids.update(df["study_id"].astype("int64").tolist())
        print(f"  {os.path.basename(p):48s} {len(df):>7d} rows")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--chexpert",
        required=True,
        help="Path to the full PhysioNet mimic-cxr-2.0.0-chexpert.csv(.gz).",
    )
    ap.add_argument(
        "--subset-dir",
        default="subset_out",
        help="Dir holding the *_imagelevel.csv subset files (default: subset_out).",
    )
    ap.add_argument(
        "--out",
        default="subset_out/chexpert_labels_subset.csv",
        help="Where to write the filtered labels (default: "
        "subset_out/chexpert_labels_subset.csv).",
    )
    args = ap.parse_args()

    if not os.path.exists(args.chexpert):
        sys.exit(f"CheXpert file not found: {args.chexpert}")

    print(f"Collecting study_ids from subset CSVs in {args.subset_dir!r}:")
    wanted = collect_subset_study_ids(args.subset_dir)
    print(f"  -> {len(wanted)} unique study_ids in subset")

    print(f"\nLoading full CheXpert labels: {args.chexpert}")
    chex = pd.read_csv(args.chexpert)
    if "study_id" not in chex.columns:
        sys.exit(
            "Expected a 'study_id' column in the CheXpert CSV. Columns found: "
            f"{list(chex.columns)}"
        )
    chex["study_id"] = chex["study_id"].astype("int64")

    filtered = chex[chex["study_id"].isin(wanted)].copy()
    matched = filtered["study_id"].nunique()
    print(
        f"  matched {matched}/{len(wanted)} subset study_ids "
        f"({matched / max(len(wanted), 1):.1%})"
    )
    if matched == 0:
        sys.exit(
            "No study_ids matched — is this the MIMIC-CXR CheXpert sheet "
            "(not the CheXpert *dataset* labels, which use a different key)?"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    filtered.to_csv(args.out, index=False)
    size_kb = os.path.getsize(args.out) / 1024
    print(
        f"\nWrote {len(filtered)} rows x {len(filtered.columns)} cols "
        f"to {args.out}  ({size_kb:.0f} KB)"
    )
    print("Commit this file so it ships to Colab with the repo.")


if __name__ == "__main__":
    main()
