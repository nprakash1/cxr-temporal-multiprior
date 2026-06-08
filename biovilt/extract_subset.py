"""
biovilt/extract_subset.py
─────────────────────────

Build a *patient-level* subset of MIMIC-CXR locally, so a collaborator on
the Stanford cluster can ship only the rows you need without breaking
prior chains.

INPUTS (you have these locally):
  --metadata  Path to mimic-cxr-2.0.0-metadata.csv  (FULL, 380k+ rows)
  --split     Path to mimic-cxr-2.0.0-split.csv     (FULL)

OUTPUTS (sent to collaborator + used later when you receive the JPGs):
  subset_metadata.csv   Filtered metadata, rows of chosen patients ONLY.
                         Same schema as the full metadata.csv.
  subset_split.csv      Filtered split.csv, same restriction.
  file_manifest.txt     Relative paths under mimic-cxr-jpg/2.0.0/ that the
                         collaborator should ship. One path per line.
                         Includes every JPG and every per-study report
                         text file for chosen patients.

THE KEY DESIGN INVARIANT:
  Once a patient is chosen, ALL of their rows are included. This is what
  keeps prior chains intact when create_dataset.py runs against the
  subset. Dropping a single intermediate study would silently re-alias
  the prior chain (see MULTIPRIOR_PLAN.md for the discussion).

EXAMPLE:
  python biovilt/extract_subset.py \
      --metadata mimic-cxr-2.0.0-metadata.csv \
      --split    mimic-cxr-2.0.0-split.csv    \
      --n-train 400 --n-val 50 --n-test 50    \
      --min-studies 3 --max-studies 12        \
      --frontal-only                          \
      --seed 42                               \
      --out-dir ./subset_out

THEN on the collaborator's cluster, from the directory that contains the
"files/" folder (i.e. inside mimic-cxr-jpg/2.0.0/):

  tar --files-from=file_manifest.txt -czf mini-mimic.tar.gz .

Or, equivalently, to rsync directly to another machine:

  rsync -av --files-from=file_manifest.txt . user@dest:/path/2.0.0/

The tarball preserves the "files/pXX/p{sid}/..." structure, so unpacking
it into your local mimic-cxr-jpg/2.0.0/ recreates a self-contained
mini-MIMIC with the same paths the existing create_dataset.py expects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATIENT-LEVEL SUBJECT SAMPLING
# ============================================================
def pick_patient_split_label(split_df: pd.DataFrame) -> pd.Series:
    """
    Each row in split.csv is keyed by (subject_id, study_id, dicom_id) plus a
    'split' column ∈ {train, validate, test}.  In practice >99% of a given
    patient's rows share the same split, but a few cross splits.

    To keep prior chains intact we have to ship the patient as a WHOLE, which
    means we need ONE split label per patient.  We use the per-patient mode
    (most common split among that patient's rows), with deterministic
    tie-breaking by lexicographic order ("test" < "train" < "validate").

    Returns: Series indexed by subject_id with value ∈ {train, validate, test}.
    """
    def _mode(g: pd.Series) -> str:
        # value_counts is sorted desc by count, then by lex order of the value
        # for ties → fully deterministic.
        return g.value_counts(sort=True).index[0]

    return split_df.groupby("subject_id")["split"].agg(_mode)


def sample_patients(
    subject_split_label: pd.Series,
    studies_per_subject: pd.Series,
    n_train: int,
    n_val: int,
    n_test: int,
    min_studies: int,
    max_studies: int,
    seed: int,
) -> list[int]:
    """
    Pick patients per split bucket subject to study-count constraints.

    `subject_split_label`  : Series subject_id -> {train, validate, test}
    `studies_per_subject`  : Series subject_id -> int  (#unique studies after
                              whatever pre-filtering was already applied to
                              the metadata table)
    """
    df = pd.DataFrame({
        "split":    subject_split_label,
        "n_studies": studies_per_subject,
    }).dropna()
    df = df[(df["n_studies"] >= min_studies) & (df["n_studies"] <= max_studies)]

    rng = np.random.default_rng(seed)

    chosen: list[int] = []
    for split_name, n_wanted in [("train", n_train),
                                 ("validate", n_val),
                                 ("test", n_test)]:
        pool = df.index[df["split"] == split_name].tolist()
        if n_wanted <= 0:
            continue
        if len(pool) < n_wanted:
            print(
                f"[warn] split={split_name!r}: requested {n_wanted} but only "
                f"{len(pool)} patients satisfy "
                f"min_studies={min_studies}, max_studies={max_studies}. "
                f"Using all available.",
                file=sys.stderr,
            )
            picks = pool
        else:
            picks = rng.choice(pool, size=n_wanted, replace=False).tolist()
        print(f"  split={split_name:8s}: pool={len(pool):6d} → picked {len(picks)}")
        chosen.extend(int(s) for s in picks)

    return sorted(set(chosen))


# ============================================================
# MANIFEST CONSTRUCTION
# ============================================================
def build_manifest(subset_meta: pd.DataFrame) -> list[str]:
    """
    Build the list of paths the collaborator must ship.  All paths are
    RELATIVE to mimic-cxr-jpg/2.0.0/ and use forward slashes.

    For every row we include:
      - the JPG image:   files/p{XX}/p{subject_id}/s{study_id}/{dicom_id}.jpg
      - the study report (deduped per study): files/p{XX}/p{subject_id}/s{study_id}.txt

    Note about reports: MIMIC-CXR-JPG by itself does NOT include reports;
    they live alongside in the MIMIC-CXR report bundle which is typically
    extracted into the same files/pXX/p{sid}/ tree. If the collaborator's
    cluster has only mimic-cxr-jpg (no reports), report lines in the
    manifest will simply be missing and `tar`/`rsync` will skip them with
    a warning. You can either:
      (a) ask the collaborator to also extract mimic-cxr-reports there,
          or
      (b) request reports separately by mailing them subset_metadata.csv
          + subset_split.csv and asking them to read each report straight
          off the cluster filesystem.
    """
    paths: list[str] = []
    seen_reports: set[str] = set()

    for _, row in subset_meta.iterrows():
        sid    = int(row["subject_id"])
        stid   = int(row["study_id"])
        dicom  = str(row["dicom_id"])
        pXX    = f"p{str(sid)[:2]}"
        psid   = f"p{sid}"
        sstid  = f"s{stid}"

        # JPG
        paths.append(f"files/{pXX}/{psid}/{sstid}/{dicom}.jpg")

        # Report — one per study, dedupe
        rpath = f"files/{pXX}/{psid}/{sstid}.txt"
        if rpath not in seen_reports:
            seen_reports.add(rpath)
            paths.append(rpath)

    # Sort for stable manifest (also makes rsync deltas friendlier).
    paths.sort()
    return paths


# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, type=Path,
                   help="Path to mimic-cxr-2.0.0-metadata.csv (full)")
    p.add_argument("--split", required=False, type=Path, default=None,
                   help="Path to mimic-cxr-2.0.0-split.csv (full). "
                        "If omitted, the script falls back to picking "
                        "--n-total patients UNIFORMLY at random (no "
                        "train/val/test stratification) and does NOT "
                        "write subset_split.csv. You'll still need "
                        "split.csv eventually to run create_dataset.py.")

    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-val",   type=int, default=50)
    p.add_argument("--n-test",  type=int, default=50)
    p.add_argument("--n-total", type=int, default=500,
                   help="Used ONLY in the --split-less fallback mode "
                        "(when --split is not provided). Number of "
                        "patients to pick uniformly at random.")


    p.add_argument("--min-studies", type=int, default=3,
                   help="Only consider patients with at least this many unique "
                        "studies after the optional --frontal-only filter. "
                        "Default 3 so every chosen patient exercises the "
                        "multi-prior path (curr + ≥2 priors).")
    p.add_argument("--max-studies", type=int, default=20,
                   help="Cap patients with too-long histories to control "
                        "total disk. Default 20.")

    p.add_argument("--frontal-only", action="store_true",
                   help="Drop rows whose ViewPosition is not in {PA, AP} "
                        "BEFORE counting per-patient studies and before "
                        "building the manifest. This roughly halves the "
                        "shipping size but means laterals will not be "
                        "available locally. The existing create_dataset.py "
                        "already filters frontals-only, so this is usually "
                        "safe.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("subset_out"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    have_split = args.split is not None

    # ---- Load ----
    print(f"[load] metadata: {args.metadata}")
    meta = pd.read_csv(args.metadata)
    print(f"       {len(meta):,} rows, {meta['subject_id'].nunique():,} patients, "
          f"{meta.groupby(['subject_id','study_id']).ngroups:,} studies")

    if have_split:
        print(f"[load] split:    {args.split}")
        split = pd.read_csv(args.split)
        print(f"       {len(split):,} rows")
    else:
        print("[load] split:    (none — running in --split-less fallback mode; "
              "will pick --n-total patients uniformly at random with no split "
              "stratification, and will NOT write subset_split.csv. You will "
              "need split.csv later to run create_dataset.py.)")
        split = None


    # ---- Optional frontal-only filter ----
    if args.frontal_only:
        before = len(meta)
        meta = meta[meta["ViewPosition"].isin(["PA", "AP"])].copy()
        print(f"[filter] frontal-only: {before:,} → {len(meta):,} rows "
              f"({meta['subject_id'].nunique():,} patients still present)")

    # ---- Per-patient stats ----
    studies_per_subject = (
        meta.groupby("subject_id")["study_id"].nunique().rename("n_studies")
    )
    print(f"[stats] patients in metadata after filter : "
          f"{len(studies_per_subject):,}")

    # ---- Pick patients ----
    if have_split:
        # Patient-level split label from split.csv (mode per patient)
        subject_split_label = pick_patient_split_label(split)
        print(f"        patients in split.csv             : "
              f"{subject_split_label.nunique():,} (label counts: "
              f"{subject_split_label.value_counts().to_dict()})")

        print(f"\n[pick] sampling patients per split "
              f"(min_studies={args.min_studies}, "
              f"max_studies={args.max_studies}, seed={args.seed}):")
        chosen = sample_patients(
            subject_split_label=subject_split_label,
            studies_per_subject=studies_per_subject,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            min_studies=args.min_studies, max_studies=args.max_studies,
            seed=args.seed,
        )
    else:
        # No-split fallback: uniform random over all eligible patients
        print(f"\n[pick] sampling {args.n_total} patients uniformly "
              f"(min_studies={args.min_studies}, "
              f"max_studies={args.max_studies}, seed={args.seed}):")
        eligible = studies_per_subject[
            (studies_per_subject >= args.min_studies)
            & (studies_per_subject <= args.max_studies)
        ].index.tolist()
        rng = np.random.default_rng(args.seed)
        if len(eligible) < args.n_total:
            print(f"[warn] only {len(eligible)} eligible patients in metadata; "
                  f"requested {args.n_total}. Using all available.",
                  file=sys.stderr)
            chosen = sorted(int(s) for s in eligible)
        else:
            chosen = sorted(
                int(s) for s in
                rng.choice(eligible, size=args.n_total, replace=False)
            )
        print(f"  eligible pool: {len(eligible):,} → picked {len(chosen)}")

    chosen_set = set(chosen)
    print(f"       total chosen patients: {len(chosen)}")

    # ---- Filter CSVs to chosen patients ----
    subset_meta = meta[meta["subject_id"].isin(chosen_set)].copy()
    print(f"\n[subset] subset_metadata: {len(subset_meta):,} rows "
          f"({subset_meta.groupby(['subject_id','study_id']).ngroups:,} studies, "
          f"{subset_meta['dicom_id'].nunique():,} images)")

    if have_split:
        subset_split = split[split["subject_id"].isin(chosen_set)].copy()
        print(f"         subset_split:    {len(subset_split):,} rows")
    else:
        subset_split = None


    # ---- Per-patient study-count distribution (sanity) ----
    dist = subset_meta.groupby("subject_id")["study_id"].nunique().value_counts().sort_index()
    print("\n[stats] subset patient study-count distribution:")
    print("        " + " | ".join(f"{int(k)} studies: {int(v)} patients" for k, v in dist.items()))

    # ---- Build manifest ----
    manifest = build_manifest(subset_meta)
    n_jpgs    = sum(1 for p in manifest if p.endswith(".jpg"))
    n_reports = sum(1 for p in manifest if p.endswith(".txt"))
    print(f"\n[manifest] {len(manifest):,} paths total "
          f"({n_jpgs:,} JPGs + {n_reports:,} reports)")
    print(f"           Rough size estimate: "
          f"{n_jpgs * 1.5 / 1024:.1f} GB JPGs (~1.5 MB/jpg avg) + ~negligible reports")

    # ---- Write outputs ----
    meta_out   = args.out_dir / "subset_metadata.csv"
    split_out  = args.out_dir / "subset_split.csv"
    manifest_out = args.out_dir / "file_manifest.txt"

    subset_meta.to_csv(meta_out,  index=False)
    if subset_split is not None:
        subset_split.to_csv(split_out, index=False)
    with open(manifest_out, "w") as f:
        f.write("\n".join(manifest) + "\n")

    print(f"\n[write] {meta_out}")
    if subset_split is not None:
        print(f"        {split_out}")
    else:
        print(f"        (skipped subset_split.csv — no --split provided)")
    print(f"        {manifest_out}")

    print("\n✅ done. Send file_manifest.txt to the collaborator and ask them "
          "to run, from inside the cluster's mimic-cxr-jpg/2.0.0/ directory:")
    print("    tar --files-from=file_manifest.txt -czf mini-mimic.tar.gz .")
    print("Then untar it into your local mimic-cxr-jpg/2.0.0/ and you can run "
          "create_dataset.py against subset_metadata.csv + subset_split.csv "
          "with perfect prior chains.")


if __name__ == "__main__":
    main()
