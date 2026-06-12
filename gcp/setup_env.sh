#!/usr/bin/env bash
# ============================================================
# GCP step 1/3 — create the conda env and install deps.
# Run ONCE per fresh instance.
#
#   bash gcp/setup_env.sh
#
# Assumes Miniconda/conda is on PATH. If not, install it first:
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
#   source $HOME/miniconda3/etc/profile.d/conda.sh
# ============================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-cxrtemporal}"
CUDA_TAG="${CUDA_TAG:-cu121}"   # match the instance's CUDA (cu118 / cu121 / ...)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] repo root : $REPO_ROOT"
echo "[setup] env name  : $ENV_NAME"
echo "[setup] cuda tag  : $CUDA_TAG"

# Create env if missing.
if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Core deps.
pip install -r requirements.txt
pip install hi-ml-multimodal

# GPU build of torch (override CPU wheel that requirements may pull on mac).
pip install --upgrade torch torchvision \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo
echo "[setup] verifying imports + GPU ..."
python - <<'PY'
import torch, torchvision, transformers, pandas, PIL, tqdm, health_multimodal
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
print("gpu count   :", torch.cuda.device_count())
print("health_multimodal: OK")
PY

echo
echo "[setup] DONE. Activate with:  conda activate $ENV_NAME"
