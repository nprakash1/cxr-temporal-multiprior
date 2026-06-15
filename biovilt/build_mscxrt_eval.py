"""
build_mscxrt_eval.py
====================

Assemble an MS-CXR-T temporal-progression EVALUATION set in the exact
multi-prior CSV format that `BioViLTDataset` already reads, so the same
model can be evaluated at K=1 (newest prior only) or K>1 (older priors
added as extra context).

INPUTS
------
--mscxrt    mscxrt_labels.csv
    Columns: patient_id, study_id_prev, study_id_curr,
             img_path_prev, img_path_curr, disease_name, comparison
    Each row = one (current, prior, pathology) progression label.
    The dicom_id is the basename of img_path_* (minus ".jpg").

--metadata  mimic-cxr-2.0.0-metadata.csv
    Columns include: dicom_id, subject_id, study_id, ViewPosition,
                     StudyDate, StudyTime
    Used to (a) order each patient's studies in time and (b) pick a
    frontal dicom for each OLDER prior study.

OUTPUT (--out)
--------------
A CSV with one row per MS-CXR-T label, in the multi-prior schema:

    split                = "test"        (so BioViLTDataset's test filter passes)
    subject_id           = patient_id
    study_id             = study_id_curr  (the CURRENT study)
    dicom_id_curr        = dicom of img_path_curr
    num_priors           = len(priors)            (>=1)
    dicom_ids_prior      = JSON list, NEWEST FIRST
    prior_study_ids      = JSON list, NEWEST FIRST
    full_report_text     = ""            (placeholder; image-only eval)

    # --- eval label columns (ignored by the model forward) ---
    pathology            = disease_name
    progression_label    = comparison
    study_id_prev        = the labeled prior study
    dicom_id_prev        = the labeled prior dicom
    num_older_priors     = num_priors - 1

PRIOR ORDERING (critical):
    priors[0] = the MS-CXR-T labeled prior  (the pair the label is about)
    priors[1..] = older frontal MIMIC studies, strictly BEFORE priors[0]'s
                  date, ordered newest-first, up to (--max-priors - 1).
At eval time, BioViLTDataset(k_max=K) truncates to the first K priors, so
K=1 reproduces the labeled pair exactly and K>1 adds older context.

USAGE
-----
    python biovilt/build_mscxrt_eval.py \
        --mscxrt   mscxrt_labels.csv \
        --metadata mimic-cxr-2.0.0-metadata.csv \
        --out      mscxrt_eval_imagelevel.csv \
        --max-priors 4
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


FRONTAL_VIEWS = {"PA", "AP"}


def _dicom_from_path(p: str) -> str:
    """img_path -> dicom_id (filename without extension)."""
    return os.path.splitext(os.path.basename(str(p)))[0]


def _time_key(study_date, study_time):
    """A sortable (date, time) key tolerant of missing/NaN StudyTime."""
    try:
        d = int(study_date)
    except (TypeError, ValueError):
        d = 0
    try:
        t = float(study_time)
    except (TypeError, ValueError):
        t = 0.0
    return (d, t)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mscxrt", default="mscxrt_labels.csv",
                    help="Path to mscxrt_labels.csv")
    ap.add_argument("--metadata", default="mimic-cxr-2.0.0-metadata.csv",
                    help="Path to mimic-cxr-2.0.0-metadata.csv")
    ap.add_argument("--out", default="mscxrt_eval_imagelevel.csv",
                    help="Where to write the assembled eval CSV.")
    ap.add_argument("--max-priors", type=int, default=4,
                    help="Max TOTAL priors to store per row (incl. the "
                         "labeled prior). The eval k_max truncates further.")
    ap.add_argument("--frontal-only", action="store_true", default=True,
                    help="Restrict OLDER priors to frontal (PA/AP) views.")
    args = ap.parse_args()

    labels = pd.read_csv(args.mscxrt)
    meta = pd.read_csv(args.metadata)

    print(f"[in] mscxrt rows   : {len(labels)}")
    print(f"[in] metadata rows : {len(meta)}")

    # ------------------------------------------------------------------
    # Index MIMIC metadata for fast per-patient timeline lookups.
    # ------------------------------------------------------------------
    meta = meta.copy()
    meta["subject_id"] = meta["subject_id"].astype(int)
    meta["study_id"] = meta["study_id"].astype(int)
    meta["_tkey"] = meta.apply(
        lambda r: _time_key(r.get("StudyDate"), r.get("StudyTime")), axis=1
    )

    # Per (subject, study) -> time key (min over that study's dicoms).
    study_time = (
        meta.groupby(["subject_id", "study_id"])["_tkey"].min().to_dict()
    )

    # Frontal dicoms only, for choosing an OLDER prior's representative image.
    frontal = meta[meta["ViewPosition"].isin(FRONTAL_VIEWS)] if args.frontal_only else meta
    # Prefer PA over AP, then earliest time, when a study has several frontals.
    frontal = frontal.assign(
        _vrank=frontal["ViewPosition"].map({"PA": 0, "AP": 1}).fillna(2)
    ).sort_values(["subject_id", "study_id", "_vrank", "_tkey"])
    # One representative frontal dicom per (subject, study).
    study_frontal_dicom = (
        frontal.groupby(["subject_id", "study_id"])["dicom_id"].first().to_dict()
    )

    # Per subject: ordered list of (tkey, study_id) over FRONTAL studies only,
    # used to pull older priors.
    subj_studies = {}
    for (sid, stid), tkey in study_time.items():
        if (sid, stid) in study_frontal_dicom:  # has a frontal image
            subj_studies.setdefault(sid, []).append((tkey, stid))
    for sid in subj_studies:
        subj_studies[sid].sort()  # ascending in time

    # ------------------------------------------------------------------
    # Build one output row per MS-CXR-T label.
    # ------------------------------------------------------------------
    out_rows = []
    n_missing_prev_time = 0
    older_hist = {}
    for _, row in labels.iterrows():
        subject_id = int(row["patient_id"])
        study_curr = int(row["study_id_curr"])
        study_prev = int(row["study_id_prev"])
        dicom_curr = _dicom_from_path(row["img_path_curr"])
        dicom_prev = _dicom_from_path(row["img_path_prev"])

        # Datetime of the labeled prior; older priors must be strictly before it.
        prev_tkey = study_time.get((subject_id, study_prev))
        if prev_tkey is None:
            n_missing_prev_time += 1

        older = []
        if prev_tkey is not None:
            for tkey, stid in subj_studies.get(subject_id, []):
                if stid in (study_prev, study_curr):
                    continue
                if tkey < prev_tkey:  # strictly older than the labeled prior
                    older.append((tkey, stid))
            # newest-first among the older studies
            older.sort(reverse=True)

        # priors: labeled prior first, then older frontal studies.
        prior_dicoms = [dicom_prev]
        prior_studies = [study_prev]
        for _tkey, stid in older:
            if len(prior_dicoms) >= args.max_priors:
                break
            prior_dicoms.append(str(study_frontal_dicom[(subject_id, stid)]))
            prior_studies.append(int(stid))

        n_older = len(prior_dicoms) - 1
        older_hist[n_older] = older_hist.get(n_older, 0) + 1

        out_rows.append({
            "split": "test",
            "subject_id": subject_id,
            "study_id": study_curr,
            "dicom_id_curr": dicom_curr,
            "num_priors": len(prior_dicoms),
            "dicom_ids_prior": json.dumps(prior_dicoms),
            "prior_study_ids": json.dumps(prior_studies),
            "full_report_text": "",  # placeholder: image-only eval
            # ---- eval label columns ----
            "pathology": row["disease_name"],
            "progression_label": row["comparison"],
            "study_id_prev": study_prev,
            "dicom_id_prev": dicom_prev,
            "num_older_priors": n_older,
        })

    out = pd.DataFrame(out_rows)
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    print(f"\n[out] wrote {len(out)} rows -> {args.out}")
    if n_missing_prev_time:
        print(f"[warn] {n_missing_prev_time} rows: labeled-prior study not "
              f"found in metadata (no older priors attached for those).")
    print("\n[dist] older priors attached per row (num_older_priors -> count):")
    for k in sorted(older_hist):
        print(f"   {k}: {older_hist[k]}")
    n_multi = int((out['num_older_priors'] >= 1).sum())
    print(f"\n[dist] rows with >=1 older prior (usable for K>1 vs K=1): {n_multi}")
    print("[dist] progression labels:")
    print(out["progression_label"].value_counts().to_string())
    print("[dist] pathologies:")
    print(out["pathology"].value_counts().to_string())


if __name__ == "__main__":
    main()
