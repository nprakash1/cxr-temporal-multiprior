import argparse
import pandas as pd
from pathlib import Path
import re
import random
import json
from tqdm import tqdm

random.seed(42)

# =======================
# CONFIG
# =======================
# Maximum number of priors to retain per study (the K_max upper bound).
# Each study row will store up to this many of its most-recent priors
# (newest first). Downstream code can choose any K ≤ K_MAX at runtime.
K_MAX = 4

# =======================
# CLI
# =======================
# Historical defaults (Stanford yunhe layout) preserved so the script still
# runs unchanged on that cluster. Override on the command line for any
# other layout, e.g. a local subset:
#
#   python biovilt/create_dataset.py \
#       --files-dir   /Users/nealprakash/Downloads/mimic_subset \
#       --metadata-csv subset_out/subset_metadata.csv \
#       --split-csv    subset_out/subset_split.csv \
#       --out-dir      subset_out
#
# Note: --files-dir is the directory that DIRECTLY contains pXX/ buckets.
# In the official MIMIC layout that's `mimic-cxr-jpg/2.0.0/files/`. If the
# `files/` wrapper was stripped during transfer (as in our `mimic_subset/`),
# just point --files-dir at the directory that holds p10/, p11/, ... .
_DEFAULT_BASE = Path("../../../../yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0")
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--files-dir",     type=Path, default=_DEFAULT_BASE / "files")
_p.add_argument("--metadata-csv",  type=Path, default=_DEFAULT_BASE / "mimic-cxr-2.0.0-metadata.csv")
_p.add_argument("--split-csv",     type=Path, default=_DEFAULT_BASE / "mimic-cxr-2.0.0-split.csv")
_p.add_argument("--out-dir",       type=Path, default=Path("."))
_p.add_argument("--k-max",         type=int,  default=K_MAX)
_p.add_argument("--save",          action="store_true",
                help="If set, write the train/val/test CSVs to --out-dir. "
                     "Without this flag the script just prints stats — "
                     "useful for dry runs.")
_args = _p.parse_args()

K_MAX     = _args.k_max
FILES_DIR = _args.files_dir
META_CSV  = _args.metadata_csv
SPLIT_CSV = _args.split_csv
OUT_DIR   = _args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TRAIN = OUT_DIR / "biovilt_pretrain_train_imagelevel.csv"
OUT_VAL   = OUT_DIR / "biovilt_pretrain_val_imagelevel.csv"
OUT_TEST  = OUT_DIR / "biovilt_pretrain_test_imagelevel.csv"

print(f"[cfg] FILES_DIR  = {FILES_DIR}")
print(f"[cfg] META_CSV   = {META_CSV}")
print(f"[cfg] SPLIT_CSV  = {SPLIT_CSV}")
print(f"[cfg] OUT_DIR    = {OUT_DIR}")
print(f"[cfg] K_MAX      = {K_MAX}")
print(f"[cfg] save CSVs  = {_args.save}")

# =======================
# LOAD METADATA
# =======================
meta = pd.read_csv(META_CSV)
split = pd.read_csv(SPLIT_CSV)


meta = meta.merge(
    split,
    on=["dicom_id", "study_id", "subject_id"],
    how="inner",
)

# IMPORTANT:
# ❌ DO NOT filter by split yet
# We must compute priors across the full patient timeline

# Frontal views only
meta = meta[meta["ViewPosition"].isin(["PA", "AP"])]

# =======================
# REPORT HANDLING
# =======================
def load_report(subject_id, study_id):
    pdir = FILES_DIR / f"p{str(subject_id)[:2]}" / f"p{subject_id}"
    rpt = pdir / f"s{study_id}.txt"
    if not rpt.exists():
        return None
    return rpt.read_text(errors="ignore")


def extract_sections(report):
    """
    Returns:
      impression_text (str or None)
      findings_text   (str or None)
    """
    report = report.replace("\r", "")

    # IMPRESSION (required)
    imp_split = re.split(r"\n\s*IMPRESSION:\s*", report, flags=re.I)
    if len(imp_split) < 2:
        return None, None
    impression = imp_split[1].strip()

    # FINDINGS (optional)
    find_split = re.split(r"\n\s*FINDINGS:\s*", report, flags=re.I)
    findings = None
    if len(find_split) > 1:
        findings = find_split[1]
        findings = findings.split("IMPRESSION:")[0].strip()

    return impression, findings


# =======================
# BUILD STUDY TABLE
# =======================
study_rows = []

groups = meta.groupby(["subject_id", "study_id"])

for (subject_id, study_id), g in tqdm(
    groups, total=groups.ngroups, desc="Parsing studies"
):
    report = load_report(subject_id, study_id)
    if report is None:
        continue

    impression, findings = extract_sections(report)
    if impression is None:
        continue

    # Randomly select ONE frontal image per study
    chosen = g.sample(n=1).iloc[0]

    # Full report (for sanity/debugging only)
    if findings is not None:
        full_report = (
            "IMPRESSION:\n" + impression +
            "\n\nFINDINGS:\n" + findings
        )
    else:
        full_report = "IMPRESSION:\n" + impression

    study_rows.append({
        "subject_id": subject_id,
        "study_id": study_id,
        "dicom_id_curr": chosen["dicom_id"],
        "StudyDate": chosen["StudyDate"],
        "StudyTime": chosen["StudyTime"],
        "split": chosen["split"],  # keep split, but don't use yet
        "impression_text": impression,
        "findings_text": findings if findings is not None else "None",
        "full_report_text": full_report,
    })

studies = pd.DataFrame(study_rows)
print(f"Valid studies: {len(studies)}")

# =======================
# DATETIME
# =======================
studies["datetime"] = pd.to_datetime(
    studies["StudyDate"].astype(str) + " " +
    studies["StudyTime"].astype(str).str.split(".").str[0],
    errors="coerce"
)

missing_dt = studies["datetime"].isnull()
studies.loc[missing_dt, "datetime"] = pd.to_datetime(
    studies.loc[missing_dt, "StudyDate"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

# =======================
# SORT & ASSIGN PRIORS  (multi-prior, up to K_MAX)
# =======================
# For each study we collect a JSON-encoded list of up to K_MAX most-recent
# priors (newest first), plus an integer `num_priors` in [0, K_MAX].
# `dicom_id_prior`, `prior_study_id`, and `has_prior` are kept as
# *backward-compat* aliases so the legacy single-prior dataset/loader
# continues to work without changes.
# =======================
studies = studies.sort_values(["subject_id", "datetime"])

studies["dicom_ids_prior"] = "[]"   # JSON list of dicom_ids (newest first)
studies["prior_study_ids"] = "[]"   # JSON list of study_ids (newest first)
studies["num_priors"]      = 0

# Backward-compat single-prior columns (always = the most-recent prior, if any)
studies["dicom_id_prior"]  = None
studies["prior_study_id"]  = None

for subject_id, g in tqdm(
    studies.groupby("subject_id"),
    desc="Assigning prior studies (up to K_MAX)"
):
    idx = g.index.tolist()
    for i in range(len(idx)):
        curr = idx[i]
        # Up to K_MAX immediate predecessors, newest-first
        prior_slice = idx[max(0, i - K_MAX): i][::-1]
        if len(prior_slice) == 0:
            continue
        dicom_ids = [str(g.loc[p, "dicom_id_curr"]) for p in prior_slice]
        study_ids = [int(g.loc[p, "study_id"]) for p in prior_slice]

        studies.at[curr, "dicom_ids_prior"] = json.dumps(dicom_ids)
        studies.at[curr, "prior_study_ids"] = json.dumps(study_ids)
        studies.at[curr, "num_priors"]     = len(dicom_ids)

        # Backward-compat singletons = newest prior
        studies.at[curr, "dicom_id_prior"] = dicom_ids[0]
        studies.at[curr, "prior_study_id"] = study_ids[0]

# =======================
# PRIOR FLAG (backward-compat)
# =======================
studies["has_prior"] = studies["num_priors"] > 0

# =======================
# FINAL TABLE
# =======================
df = studies[[
    "dicom_id_curr",
    "subject_id",
    "study_id",

    # New multi-prior columns (JSON-encoded lists, newest first)
    "dicom_ids_prior",
    "prior_study_ids",
    "num_priors",

    # Backward-compat single-prior columns (= newest prior)
    "dicom_id_prior",
    "prior_study_id",
    "has_prior",

    "impression_text",
    "findings_text",
    "full_report_text",
    "split",
]]

# =======================
# SPLIT & SAVE
# =======================
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "validate"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)

if _args.save:
    train_df.to_csv(OUT_TRAIN, index=False)
    val_df.to_csv(OUT_VAL, index=False)
    test_df.to_csv(OUT_TEST, index=False)
    print(f"\n[write] {OUT_TRAIN}")
    print(f"        {OUT_VAL}")
    print(f"        {OUT_TEST}")
else:
    print("\n[dry-run] --save not set; CSVs NOT written. "
          "Re-run with --save to persist them.")


# =======================
# STATS
# =======================
print("\n=== IMAGE-LEVEL COUNTS ===")
print(f"Train rows: {len(train_df)}")
print(f"Val rows:   {len(val_df)}")
print(f"Test rows:  {len(test_df)}")

print("\n=== PRIOR FRACTIONS (has_prior, backward-compat) ===")
print(f"Train w/ priors: {train_df['has_prior'].mean():.3f}")
print(f"Val w/ priors:   {val_df['has_prior'].mean():.3f}")
print(f"Test w/ priors:  {test_df['has_prior'].mean():.3f}")

print(f"\n=== NUM_PRIORS DISTRIBUTION (K_MAX = {K_MAX}) ===")
for name, sub in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    dist = sub["num_priors"].value_counts().sort_index()
    pretty = " | ".join(f"K={k}: {n}" for k, n in dist.items())
    print(f"{name:5s} {pretty}")
