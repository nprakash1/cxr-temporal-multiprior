# Running `cxr-temporal` on a GCP instance

These three scripts package the full workflow for a GCP VM that **already has
the MIMIC-CXR-JPG files on disk** (the usual case for a credentialed PhysioNet
mirror). No images are copied; you only point the scripts at the data.

```
gcp/
├── setup_env.sh      # 1. conda env + deps (run once)
├── build_dataset.sh  # 2. generate train/val/test CSVs from on-disk MIMIC
└── run_train.sh      # 3. launch DDP training with torchrun
```

## Prerequisites on the instance

- A GPU VM (e.g. `a2-highgpu-1g` = 1×A100, or `-4g` = 4×A100) with NVIDIA
  drivers + CUDA installed (`nvidia-smi` works).
- `conda` available (Miniconda). If not, install it — see the header of
  `setup_env.sh`.
- The MIMIC-CXR-JPG v2.0.0 tree on disk. You need:
  - the image root: the directory whose direct children are `p10/ … p19/`
    (official layout: `.../mimic-cxr-jpg/2.0.0/files`),
  - `mimic-cxr-2.0.0-metadata.csv`,
  - `mimic-cxr-2.0.0-split.csv`,
  - `mimic-cxr-2.0.0-chexpert.csv` (only if you want to drop "No Finding"),
  - the per-study report `.txt` files (under `pXX/pSUBJECT/sSTUDY.txt`).

Find the image root if unsure:
```bash
find / -maxdepth 8 -type d -name files -path '*mimic-cxr-jpg*' 2>/dev/null
```

## 0. Clone the repo
```bash
git clone https://github.com/nprakash1/cxr-temporal-multiprior.git
cd cxr-temporal-multiprior
```

## 1. Environment (once)
```bash
# CUDA_TAG must match the instance (cu118 / cu121 / ...)
CUDA_TAG=cu121 bash gcp/setup_env.sh
conda activate cxrtemporal
```

## 2. Build the dataset CSVs
The committed `subset_out/` CSVs only cover ~2k studies. On GCP you have the
full data, so regenerate at full scale (this is what fixes overfitting).

```bash
DATA=/path/to/mimic-cxr-jpg/2.0.0

# Full MIMIC (keep everything):
FILES_DIR=$DATA/files \
META_CSV=$DATA/mimic-cxr-2.0.0-metadata.csv \
SPLIT_CSV=$DATA/mimic-cxr-2.0.0-split.csv \
K_MAX=4 OUT_DIR=full_out \
bash gcp/build_dataset.sh

# …or drop all "No Finding" studies (lowers batch collision rate):
FILES_DIR=$DATA/files \
META_CSV=$DATA/mimic-cxr-2.0.0-metadata.csv \
SPLIT_CSV=$DATA/mimic-cxr-2.0.0-split.csv \
CHEXPERT_CSV=$DATA/mimic-cxr-2.0.0-chexpert.csv \
DROP_NO_FINDING=1 K_MAX=4 OUT_DIR=full_out_nonf \
bash gcp/build_dataset.sh
```
This writes `biovilt_pretrain_{train,val,test}_imagelevel.csv` (+ a
`combined` symlink for the validation loader) into `$OUT_DIR`.

## 3. Train
```bash
DATA=/path/to/mimic-cxr-jpg/2.0.0

# K=4 multi-prior on the full set:
IMAGE_ROOT=$DATA/files \
CSV_DIR=full_out \
K_MAX=4 BATCH_SIZE=32 EPOCHS=50 \
bash gcp/run_train.sh

# K=1 baseline (newest prior only):
IMAGE_ROOT=$DATA/files CSV_DIR=full_out K_MAX=1 bash gcp/run_train.sh
```
Checkpoints → `checkpoints/epoch_*.pt` + `best.pt`; metrics →
`logs/val_metrics.csv`.

## Knobs (env vars)

| var | default | meaning |
|---|---|---|
| `ENV_NAME` | `cxrtemporal` | conda env name |
| `CUDA_TAG` | `cu121` | torch CUDA wheel tag (setup only) |
| `FILES_DIR` | — (required) | dir containing `pXX/` (dataset build) |
| `META_CSV` / `SPLIT_CSV` | — (required) | MIMIC metadata + split csv |
| `CHEXPERT_CSV` | — | needed only with `DROP_NO_FINDING=1` |
| `DROP_NO_FINDING` | `0` | `1` = remove "No Finding" studies |
| `K_MAX` | `4` | max priors (build) / k-max (train) |
| `OUT_DIR` | `full_out` | where CSVs are written |
| `IMAGE_ROOT` | — (required, train) | same as `FILES_DIR`, for training |
| `CSV_DIR` | `full_out` | dir of the generated CSVs |
| `BATCH_SIZE` | `32` | per-GPU batch size |
| `EPOCHS` | `50` | training epochs |
| `NGPU` | auto | GPUs for DDP (`torch.cuda.device_count()`) |
| `NUM_WORKERS` | `8` | DataLoader workers |
| `CHECKPOINT_DIR` / `LOG_DIR` | `checkpoints/` / `logs/` | outputs |

## Notes & gotchas

- **Fresh vs resume:** if `CHECKPOINT_DIR` already contains `epoch_*.pt`,
  `resume_train.py` auto-resumes the latest. Use an empty checkpoint dir for
  a clean run (the script prints a warning).
- **Keep the GPU fed:** set `NUM_WORKERS` near the instance's vCPU count.
- **Persist results:** GCP boot disks can be ephemeral — copy
  `checkpoints/best.pt` and `logs/val_metrics.csv` to a GCS bucket
  (`gsutil cp ...`) when done.
- **Long runs:** launch under `tmux`/`nohup` so an SSH drop doesn't kill it.
