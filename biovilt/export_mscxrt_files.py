"""
export_mscxrt_files.py
======================

Produce everything a collaborator needs to run the MS-CXR-T evaluation on
another machine WITHOUT shipping all of MIMIC:

  1. The exact set of images referenced by the eval CSV (currents + every
     prior, deduplicated).
  2. A metadata manifest of ONLY those images, in MIMIC-CXR metadata CSV
     format (same columns as mimic-cxr-2.0.0-metadata.csv) — so the bundle
     is self-describing.
  3. A relative file list (pXX/pSUBJECT/sSTUDY/dicom.jpg) you can hand to
     rsync/tar to copy just those JPGs, preserving the MIMIC directory
     layout that `dataset.resolve_image_path` expects.

It also PRINTS ready-to-run rsync/tar commands.

INPUTS
------
--eval-csv   mscxrt_eval_imagelevel.csv   (output of build_mscxrt_eval.py)
--metadata   mimic-cxr-2.0.0-metadata.csv
--files-root /path/to/mimic-cxr-jpg/2.0.0/files   (the cluster's files/ dir)
--out-dir    export_mscxrt

OUTPUTS (in --out-dir)
----------------------
  mscxrt_files_metadata.csv   metadata rows for the needed dicoms (MIMIC format)
  mscxrt_files_list.txt       relative JPG paths, one per line (for rsync/tar)

USAGE
-----
    python biovilt/export_mscxrt_files.py \
        --eval-csv   mscxrt_eval_imagelevel.csv \
        --metadata   mimic-cxr-2.0.0-metadata.csv \
        --files-root /scratch/.../mimic-cxr-jpg/2.0.0/files \
        --out-dir    export_mscxrt

Then copy the JPGs (preserving structure):
    rsync -av --files-from=export_mscxrt/mscxrt_files_list.txt \
        /scratch/.../files/  ./mscxrt_bundle/files/

NOTE: the images + manifest are credentialed PhysioNet data. Transfer only
via a private channel to someone who holds the MIMIC-CXR DUA.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def _rel_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    pid = str(subject_id)
    return f"p{pid[:2]}/p{pid}/s{study_id}/{dicom_id}.jpg"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--eval-csv", default="mscxrt_eval_imagelevel.csv")
    ap.add_argument("--metadata", default="mimic-cxr-2.0.0-metadata.csv")
    ap.add_argument("--files-root", default=None,
                    help="MIMIC files/ dir on this machine (for the copy "
                         "command + optional on-disk existence check).")
    ap.add_argument("--out-dir", default="export_mscxrt")
    args = ap.parse_args()

    ev = pd.read_csv(args.eval_csv)
    meta = pd.read_csv(args.metadata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect every (subject, study, dicom) the eval set references:
    # the current image + all priors in each row.
    # ------------------------------------------------------------------
    needed = set()  # (subject_id, study_id, dicom_id)
    for _, r in ev.iterrows():
        sid = int(r["subject_id"])
        needed.add((sid, int(r["study_id"]), str(r["dicom_id_curr"])))
        p_studies = json.loads(r["prior_study_ids"])
        p_dicoms = json.loads(r["dicom_ids_prior"])
        for st, d in zip(p_studies, p_dicoms):
            needed.add((sid, int(st), str(d)))

    print(f"[in] eval rows            : {len(ev)}")
    print(f"[in] unique images needed : {len(needed)}")

    needed_dicoms = {d for (_, _, d) in needed}

    # ------------------------------------------------------------------
    # 1) Metadata manifest (MIMIC format) for exactly those dicoms.
    # ------------------------------------------------------------------
    manifest = meta[meta["dicom_id"].astype(str).isin(needed_dicoms)].copy()
    meta_out = out_dir / "mscxrt_files_metadata.csv"
    manifest.to_csv(meta_out, index=False)
    print(f"[out] metadata manifest    : {meta_out}  ({len(manifest)} rows)")

    missing_in_meta = len(needed_dicoms) - manifest["dicom_id"].astype(str).nunique()
    if missing_in_meta:
        print(f"[warn] {missing_in_meta} needed dicoms NOT found in metadata!")

    # ------------------------------------------------------------------
    # 2) Relative file list for rsync/tar (preserves MIMIC layout).
    # ------------------------------------------------------------------
    rel_paths = sorted(_rel_path(s, st, d) for (s, st, d) in needed)
    list_out = out_dir / "mscxrt_files_list.txt"
    list_out.write_text("\n".join(rel_paths) + "\n")
    print(f"[out] file list            : {list_out}  ({len(rel_paths)} paths)")

    # ------------------------------------------------------------------
    # Optional: on-disk existence check (only if --files-root given).
    # ------------------------------------------------------------------
    if args.files_root:
        root = Path(args.files_root)
        missing = [p for p in rel_paths if not (root / p).exists()]
        print(f"[disk] checked {len(rel_paths)} files under {root}")
        print(f"[disk] missing on disk     : {len(missing)}")
        if missing:
            miss_out = out_dir / "mscxrt_missing_on_disk.txt"
            miss_out.write_text("\n".join(missing) + "\n")
            print(f"[disk] wrote missing list  : {miss_out}")

    # ------------------------------------------------------------------
    # Print copy commands.
    # ------------------------------------------------------------------
    root_str = args.files_root or "/path/to/mimic-cxr-jpg/2.0.0/files"
    print("\n# ---- copy the JPGs, preserving MIMIC layout ----")
    print(f"rsync -av --files-from={list_out} \\")
    print(f"    {root_str}/ ./mscxrt_bundle/files/")
    print("\n# ---- or as a single tarball ----")
    print(f"tar -C {root_str} -czf mscxrt_bundle.tar.gz -T {list_out}")
    print("\n# Then ship ./mscxrt_bundle/ (files/ + manifest + the eval CSV)")
    print("# via a PRIVATE channel to a DUA-holder. The recipient points")
    print("# the evaluator's --image-root at the copied files/ dir.")


if __name__ == "__main__":
    main()
