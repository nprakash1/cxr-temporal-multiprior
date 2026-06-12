# Running on another computer (e.g. a GCP GPU instance)

This guide is everything needed to train the BioViL-T multi-prior model on a
fresh machine. **You provide** the GPU box + the credentialed MIMIC data;
**git provides** the code; then you run: *filter normals → build CSVs → train*.

---

## 0. Prerequisites on the target machine

- **A GPU** with NVIDIA drivers + CUDA installed (`nvidia-smi` must work).
- **Python + conda** (Miniconda is fine).
- **The credentialed MIMIC-CXR-JPG v2.0.0 dataset on local disk.** This is
  downloaded from PhysioNet with *your* credentials — it is **not** in this
  repo and must never be committed. You need:
  - `files/` — the images (`pXX/pSUBJECT/sSTUDY/*.jpg`)
  - `mimic-cxr-2.0.0-metadata.csv` — per-image metadata
  - `mimic-cxr-2.0.0-chexpert.csv` — per-study disease labels
  - `mimic-cxr-2.0.0-split.csv` — official train/validate/test split
  - the report `.txt` files (`files/pXX/pSUBJECT/sSTUDY.txt`)

> Nothing in the code knows about GCP — it is plain local-filesystem I/O. You
> simply point the flags at wherever MIMIC lives on this machine.

---

## 1. Get the code

```bash
git clone <your-repo-url>
cd cxr-temporal
```

---

## 2. Install dependencies

```bash
conda create -n cxrtemporal python=3.10 -y
conda activate cxrtemporal

pip install -r requirements.txt
pip install hi-ml-multimodal

# GPU build of torch — match the machine's CUDA (cu121 shown; use cu118 etc.)
pip install --upgrade torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"
# -> cuda: True gpus: N
```

---

## 3. Point at the data

```bash
DATA=/path/to/mimic-cxr-jpg/2.0.0      # wherever MIMIC lives on THIS machine
```

Not sure where it is?

```bash
find / -name 'mimic-cxr-2.0.0-metadata.csv' 2>/dev/null
find / -type d -name files -path '*mimic-cxr-jpg*' 2>/dev/null
```

---

## 4. Remove "No Finding" studies

Drops every study CheXpert labels `No Finding == 1.0` (~38% of images), which
removes the dominant duplicate "normal" fingerprint that hurts contrastive
training. Output is a filtered copy of the metadata — **no split happens here.**

```bash
mkdir -p full_out_nonf
python biovilt/filter_no_finding_metadata.py \
  --metadata $DATA/mimic-cxr-2.0.0-metadata.csv \
  --chexpert $DATA/mimic-cxr-2.0.0-chexpert.csv \
  --out      full_out_nonf/metadata-no_no_finding.csv
```

> To train on the **full** dataset instead (keep normals), skip this step and
> use `$DATA/mimic-cxr-2.0.0-metadata.csv` directly in step 5.

---

## 5. Build the train/val/test CSVs

Merges the (filtered) metadata with MIMIC's official `split.csv`, keeps frontal
(PA/AP) views, parses reports, assigns up to `K_MAX` temporal priors per study,
and writes the image-level CSVs. The split is **inherited from MIMIC**, not
recomputed.

```bash
python biovilt/create_dataset.py \
  --files-dir    $DATA/files \
  --metadata-csv full_out_nonf/metadata-no_no_finding.csv \
  --split-csv    $DATA/mimic-cxr-2.0.0-split.csv \
  --out-dir      full_out_nonf \
  --k-max        4 \
  --save

# the validation loader expects a "combined" filename — add the symlink:
ln -sf biovilt_pretrain_val_imagelevel.csv \
       full_out_nonf/biovilt_pretrain_combined_imagelevel.csv
```

This produces in `full_out_nonf/`:
`biovilt_pretrain_{train,val,test}_imagelevel.csv` (+ the combined symlink).

> `create_dataset.py` has a hard-coded Stanford default path, so on any other
> machine you **must** pass `--files-dir`, `--metadata-csv`, and `--split-csv`.

---

## 6. Train

```bash
NUM_WORKERS=8 torchrun --nproc_per_node=<NUM_GPUS> biovilt/resume_train.py \
  --image-root      $DATA/files \
  --csv-dir         full_out_nonf \
  --checkpoint-dir  checkpoints \
  --log-dir         logs \
  --k-max 4 --mode biovilt --batch-size 32 --epochs 50
```

- `<NUM_GPUS>` = number of GPUs (`1`, `4`, …) — a **torchrun** argument.
- `--k-max 1` = newest-prior baseline (BioViL-T); `--k-max 4` = multi-prior.
- Outputs: `checkpoints/epoch_*.pt` + `best.pt`; metrics in `logs/val_metrics.csv`.

> **What you get:** training itself produces no graphs — just scrolling console
> logs, the `.pt` checkpoints, and `logs/val_metrics.csv` (one row per epoch:
> `epoch,val_total,val_global,val_local,val_mlm`).

---

## 7. Plot the loss curves (optional)

Turn `val_metrics.csv` into PNG graphs. This is a separate, GPU-free step you
can run during or after training:

```bash
python biovilt/plot_metrics.py --csv logs/val_metrics.csv --out-dir logs
```

Writes into `--out-dir`:
- `val_loss_total.png` — total validation loss vs epoch (best epoch marked)
- `val_loss_components.png` — global / local / mlm losses on one axis

Uses a headless matplotlib backend, so it works over SSH on the GCP box. Add
`--show` only if you have a display. Pull the PNGs back with
`gcloud compute scp` (or `gsutil cp`).

---


## Command-line flags for `resume_train.py`

| Flag | Default | Meaning |
|---|---|---|
| `--image-root` | cluster default | dir that directly contains `pXX/` |
| `--csv-dir` | cluster default | dir holding the generated CSVs |
| `--train-csv` / `--val-csv` | from `--csv-dir` | explicit CSV paths |
| `--checkpoint-dir` | cluster default | where checkpoints are written |
| `--log-dir` | cluster default | where `val_metrics.csv` is written |
| `--k-max` | `1` | priors per sample (1 = newest only) |
| `--mode` | `biovilt` | init weights: `biovil` / `biovilt` / `biovilt_finetuned` |
| `--epochs` | `50` | training epochs |
| `--batch-size` | `32` | **per-GPU** batch size |
| `--resume` | `None` | continue from a saved `epoch_N.pt` |
| `--init-from` | `None` | fresh start from raw upstream weights |

**Not** flags (set elsewhere):
- **GPU count** → `torchrun --nproc_per_node=N`
- **`NUM_WORKERS`** → env var (default 8), e.g. `NUM_WORKERS=8 torchrun ...`
- **`LR=2e-5`, `WEIGHT_DECAY=0.01`, `WARMUP_RATIO=0.03`, loss weights
  `W_GLOBAL=1.0 / W_LOCAL=0.5 / W_MLM=1.0`** → hard-coded in
  `resume_train.py` (edit the file to change)

---

## Notes & gotchas

- **Fresh vs resume:** if `checkpoint-dir` already contains `epoch_*.pt`, the
  trainer auto-resumes the latest. Use an **empty** checkpoint dir for a clean
  run, or pass `--init-from`.
- **Long jobs:** launch under `tmux` or `nohup` so an SSH drop doesn't kill it.
- **Keep the GPU fed:** set `NUM_WORKERS` near the machine's vCPU count.
- **Persist results:** copy `checkpoints/best.pt` and `logs/val_metrics.csv`
  off the instance (e.g. `gsutil cp ... gs://your-bucket/`) — boot disks can
  be ephemeral.

---

## Credentialed-data rules (important)

Never commit or push any of these — they are credentialed PhysioNet data or
derived from it (the repo's `.gitignore` already blocks them):

- `mimic-cxr-2.0.0-*.csv` (metadata, chexpert, split)
- `*-no_no_finding.csv` (filtered metadata)
- `biovilt_pretrain_*_imagelevel.csv` and `subset_out/` (contain report text)
- model weights (`*.pt`)

To move a processed file between machines, use a **private** channel
(`gcloud compute scp`, or `gsutil cp` via your own private bucket) — **not**
a git remote. Easiest of all: just regenerate it on the target machine
(steps 4–5), since the source CSVs already live there.
