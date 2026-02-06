import pandas as pd
from pathlib import Path
import re
import random
from tqdm import tqdm

random.seed(42)

# =======================
# PATHS
# =======================
BASE = Path("../../../../yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0")
FILES_DIR = BASE / "files"

META_CSV = BASE / "mimic-cxr-2.0.0-metadata.csv"
SPLIT_CSV = BASE / "mimic-cxr-2.0.0-split.csv"

OUT_TRAIN = "biovilt_pretrain_train_imagelevel.csv"
OUT_VAL   = "biovilt_pretrain_val_imagelevel.csv"
OUT_TEST  = "biovilt_pretrain_test_imagelevel.csv"

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
# SORT & ASSIGN PRIORS (CRITICAL FIX)
# =======================
studies = studies.sort_values(["subject_id", "datetime"])

studies["dicom_id_prior"] = None
studies["prior_study_id"] = None

for subject_id, g in tqdm(
    studies.groupby("subject_id"),
    desc="Assigning prior studies"
):
    idx = g.index.tolist()
    for i in range(1, len(idx)):
        prev = idx[i - 1]
        curr = idx[i]

        studies.at[curr, "dicom_id_prior"] = g.loc[prev, "dicom_id_curr"]
        studies.at[curr, "prior_study_id"] = g.loc[prev, "study_id"]

# =======================
# PRIOR FLAG
# =======================
studies["has_prior"] = studies["dicom_id_prior"].notnull()

# =======================
# FINAL TABLE
# =======================
df = studies[[
    "dicom_id_curr",
    "dicom_id_prior",
    "subject_id",
    "study_id",
    "prior_study_id",   # ✅ REQUIRED
    "impression_text",
    "findings_text",
    "full_report_text",
    "has_prior",
    "split",
]]

# =======================
# SPLIT & SAVE
# =======================
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "validate"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)

train_df.to_csv(OUT_TRAIN, index=False)
val_df.to_csv(OUT_VAL, index=False)
test_df.to_csv(OUT_TEST, index=False)

# =======================
# STATS
# =======================
print("\n=== IMAGE-LEVEL COUNTS ===")
print(f"Train rows: {len(train_df)}")
print(f"Val rows:   {len(val_df)}")
print(f"Test rows:  {len(test_df)}")

print("\n=== PRIOR FRACTIONS ===")
print(f"Train w/ priors: {train_df['has_prior'].mean():.3f}")
print(f"Val w/ priors:   {val_df['has_prior'].mean():.3f}")
print(f"Test w/ priors:  {test_df['has_prior'].mean():.3f}")

