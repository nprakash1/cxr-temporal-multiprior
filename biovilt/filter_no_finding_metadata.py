"""Cross-reference mimic-cxr-2.0.0-metadata.csv against
mimic-cxr-2.0.0-chexpert.csv and drop every image whose STUDY is labeled
"No Finding" (the CheXpert `No Finding` column == 1.0).

Rationale: ~33% of MIMIC studies are "No Finding" normals. They collapse
to one identical disease fingerprint, which dominates contrastive batches
and creates false negatives. Removing them de-duplicates the dominant
profile and lowers the measured collision rate.

Keys:
  metadata : dicom_id, subject_id, study_id, ... (one row per IMAGE, ~377k)
  chexpert : subject_id, study_id, + 14 disease cols (one row per STUDY, ~227k)

A study is dropped if its CheXpert `No Finding` cell == 1.0. Studies with
no CheXpert row (unlabeled) are KEPT (they were never flagged normal).

Usage:
    python biovilt/filter_no_finding_metadata.py \
        --metadata mimic-cxr-2.0.0-metadata.csv \
        --chexpert mimic-cxr-2.0.0-chexpert.csv \
        --out      mimic-cxr-2.0.0-metadata-no_no_finding.csv
"""
import argparse
import os
import sys

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", default="mimic-cxr-2.0.0-metadata.csv",
                    help="Path to mimic-cxr-2.0.0-metadata.csv (per-image).")
    ap.add_argument("--chexpert", default="mimic-cxr-2.0.0-chexpert.csv",
                    help="Path to mimic-cxr-2.0.0-chexpert.csv (per-study labels).")
    ap.add_argument("--out", default="mimic-cxr-2.0.0-metadata-no_no_finding.csv",
                    help="Where to write the filtered metadata CSV.")
    ap.add_argument("--no-finding-col", default="No Finding",
                    help="Name of the CheXpert No-Finding column.")
    args = ap.parse_args()

    for p in (args.metadata, args.chexpert):
        if not os.path.exists(p):
            sys.exit(f"File not found: {p}")

    meta = pd.read_csv(args.metadata)
    chex = pd.read_csv(args.chexpert)

    for col in ("subject_id", "study_id"):
        if col not in meta.columns:
            sys.exit(f"metadata missing column {col!r}")
        if col not in chex.columns:
            sys.exit(f"chexpert missing column {col!r}")
        meta[col] = meta[col].astype("int64")
        chex[col] = chex[col].astype("int64")

    if args.no_finding_col not in chex.columns:
        sys.exit(f"chexpert missing column {args.no_finding_col!r}. "
                 f"Columns: {list(chex.columns)}")

    # Studies flagged "No Finding" == 1.0.
    nf_mask = chex[args.no_finding_col].fillna(0.0) == 1.0
    nf_studies = set(zip(chex.loc[nf_mask, "subject_id"],
                         chex.loc[nf_mask, "study_id"]))

    meta_keys = list(zip(meta["subject_id"], meta["study_id"]))
    drop_mask = pd.Series([k in nf_studies for k in meta_keys], index=meta.index)

    kept = meta[~drop_mask].copy()

    # Reporting
    n_total = len(meta)
    n_drop = int(drop_mask.sum())
    n_keep = len(kept)
    n_studies_total = meta[["subject_id", "study_id"]].drop_duplicates().shape[0]
    n_studies_keep = kept[["subject_id", "study_id"]].drop_duplicates().shape[0]
    print(f"CheXpert 'No Finding'==1 studies : {len(nf_studies):>7d}")
    print(f"metadata images total           : {n_total:>7d}")
    print(f"  dropped (No Finding studies)  : {n_drop:>7d}  ({n_drop / n_total:.1%})")
    print(f"  kept                          : {n_keep:>7d}  ({n_keep / n_total:.1%})")
    print(f"metadata studies total          : {n_studies_total:>7d}")
    print(f"  kept                          : {n_studies_keep:>7d}")

    kept.to_csv(args.out, index=False)
    size_mb = os.path.getsize(args.out) / 1024**2
    print(f"\nWrote {n_keep} rows x {len(kept.columns)} cols to {args.out}  "
          f"({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
