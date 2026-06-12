#!/usr/bin/env bash
# ============================================================
# GCP step 3/3 — launch training with torchrun (DDP over all GPUs).
#
#   IMAGE_ROOT=/data/mimic-cxr-jpg/2.0.0/files \
#   CSV_DIR=full_out \
#   K_MAX=4 \
#   bash gcp/run_train.sh
#
# Override anything via env vars (see defaults below).
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-cxrtemporal}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---- Data / output paths (exported so resume_train.py picks them up) ----
export IMAGE_ROOT="${IMAGE_ROOT:?set IMAGE_ROOT to the dir that directly contains pXX/ folders}"
export CSV_DIR="${CSV_DIR:-$REPO_ROOT/full_out}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints}"
export LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

# ---- Training hyperparams (CLI flags to resume_train.py) ----
K_MAX="${K_MAX:-4}"
MODE="${MODE:-biovilt}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-50}"

# ---- GPU count: auto-detect, override with NGPU ----
NGPU="${NGPU:-$(python -c 'import torch;print(torch.cuda.device_count())')}"
MASTER_PORT="${MASTER_PORT:-29501}"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

echo "[train] IMAGE_ROOT     = $IMAGE_ROOT"
echo "[train] CSV_DIR        = $CSV_DIR"
echo "[train] CHECKPOINT_DIR = $CHECKPOINT_DIR"
echo "[train] LOG_DIR        = $LOG_DIR"
echo "[train] K_MAX/MODE     = $K_MAX / $MODE"
echo "[train] BATCH/EPOCHS   = $BATCH_SIZE / $EPOCHS"
echo "[train] NGPU           = $NGPU"

# Warn if resuming would be triggered accidentally.
if compgen -G "$CHECKPOINT_DIR/epoch_*.pt" >/dev/null; then
  echo "[train] NOTE: $CHECKPOINT_DIR already has epoch_*.pt — resume_train.py"
  echo "        will AUTO-RESUME the latest. Use an empty CHECKPOINT_DIR for a fresh run."
fi

torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  biovilt/resume_train.py \
  --k-max      "$K_MAX" \
  --mode       "$MODE" \
  --batch-size "$BATCH_SIZE" \
  --epochs     "$EPOCHS"
