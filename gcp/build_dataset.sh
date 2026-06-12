#!/usr/bin/env bash
# ============================================================
# GCP step 2/3 — build the train/val/test CSVs from the MIMIC files
# already on the instance, then (optionally) drop "No Finding" studies.
#
#   FILES_DIR=/data/mimic-cxr-jpg/2.0.0/files \
#   META_CSV=/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv \
#   SPLIT_CSV=/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv \
#   CHEXPERT_CSV=/data/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv \
#   DROP_NO_FINDING=1 \
#   bash gcp/build_dataset.sh
#
# Output CSVs land in $OUT_DIR (default: full_out/).
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- Required: point these at the on-disk MIMIC files ----
FILES_DIR="${FILES_DIR:?set FILES_DIR to the dir that directly contains pXX/ folders}"
META_CSV="${META_CSV:?set META_CSV to mimic-cxr-2.0.0-metadata.csv}"
SPLIT_CSV="${SPLIT_CSV:?set SPLIT_CSV to mimic-cxr-2.0.0-split.csv}"

# ---- Optional ----
CHEXPERT_CSV="${CHEXPERT_CSV:-}"      # needed only if DROP_NO_FINDING=1
DROP_NO_FINDING="${DROP_NO_FINDING:-0}"
K_MAX="${K_MAX:-4}"
OUT_DIR="${OUT_DIR:-full_out}"

ENV_NAME="${ENV_NAME:-cxrtemporal}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$OUT_DIR"

# ---- Optionally pre-filter the metadata to remove No-Finding studies ----
# create_dataset.py only ever sees rows in META_CSV, so dropping No-Finding
# studies here removes them from every generated train/val/test CSV.
META_TO_USE="$META_CSV"
if [[ "$DROP_NO_FINDING" == "1" ]]; then
  : "${CHEXPERT_CSV:?DROP_NO_FINDING=1 requires CHEXPERT_CSV}"
  FILTERED="$OUT_DIR/metadata-no_no_finding.csv"
  echo "[build] dropping No-Finding studies -> $FILTERED"
  python biovilt/filter_no_finding_metadata.py \
    --metadata "$META_CSV" \
    --chexpert "$CHEXPERT_CSV" \
    --out      "$FILTERED"
  META_TO_USE="$FILTERED"
fi

echo "[build] generating CSVs (K_MAX=$K_MAX) from $META_TO_USE"
python biovilt/create_dataset.py \
  --files-dir    "$FILES_DIR" \
  --metadata-csv "$META_TO_USE" \
  --split-csv    "$SPLIT_CSV" \
  --out-dir      "$OUT_DIR" \
  --k-max        "$K_MAX" \
  --save

# create_dataset.py writes train/val/test separately, but resume_train.py
# looks for *_combined_imagelevel.csv for validation. Symlink val -> combined.
ln -sf "biovilt_pretrain_val_imagelevel.csv" \
       "$OUT_DIR/biovilt_pretrain_combined_imagelevel.csv"

echo
echo "[build] DONE. CSVs in: $OUT_DIR"
ls -1 "$OUT_DIR"/biovilt_pretrain_*.csv
