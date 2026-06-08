# cxr-temporal-multiprior

Multi-prior temporal generalization of BioViL-T for chest X-ray
representation learning. Generalizes the upstream K=1 single-prior
joint self-attention to arbitrary K ∈ {0, 1, …, K_max} priors per
sample, with full backward compatibility and an automatic checkpoint
migration utility for the official BioViL-T weights.

[![Verify in Colab](https://colab.research.google.com/assets/colab-badge.svg) — **verify**](https://colab.research.google.com/github/nprakash1/cxr-temporal-multiprior/blob/main/colab/run_on_colab.ipynb)
&nbsp;&nbsp;
[![Train in Colab](https://colab.research.google.com/assets/colab-badge.svg) — **train**](https://colab.research.google.com/github/nprakash1/cxr-temporal-multiprior/blob/main/colab/train_on_colab.ipynb)

## Quick start

### Colab — verify the build (no data needed)

`colab/run_on_colab.ipynb` clones the repo, installs deps, and runs
the full 35-test verification suite plus a `K_max=4` forward-pass
demo on a T4 runtime. ~5 min, no MIMIC-CXR required.

### Colab — actually train on a MIMIC subset

`colab/train_on_colab.ipynb` takes a single zip of MIMIC-CXR-JPG
patient folders (`p10/ … p19/` layout) uploaded to your Drive,
unzips it to `/content`, and launches single-GPU training via
`torchrun --nproc_per_node=1` with the cluster paths overridden by
env vars. **The only thing you need to upload is the zip** — the
CSVs ship with the repo and the upstream BioViL-T weights are
auto-downloaded by `hi-ml`.

### Local

```bash
git clone https://github.com/nprakash1/cxr-temporal-multiprior.git
cd cxr-temporal-multiprior
conda create -n cxrtemporal python=3.10 -y && conda activate cxrtemporal
pip install -r requirements.txt
pip install hi-ml-multimodal

# Verify the build (35 tests, ~30s on CPU):
python biovilt/test_smoke.py        # 32 architectural invariant tests
python biovilt/test_migration.py    # 3 end-to-end migration tests
```

## What's in the box

| Path | Purpose |
|---|---|
| `biovilt/tempcxr/modules/multi_prior_block.py` | `MultiPriorTransformerPooler` — joint self-attention over `(K+1)·L` tokens with extended `type_embed_multi` and `key_padding_mask`. The only multi-prior fusion path. |
| `biovilt/tempcxr/modules/image_encoder.py` | `BioViLTImageEncoder` — single CNN pass over `(B*(K+1), 3, H, W)`, then routes to the multi-prior pooler. |
| `biovilt/tempcxr/modules/tempcxr_model.py` | `TempCXR` — public API: `forward(curr, prior_imgs, prior_mask, texts)`. |
| `biovilt/dataset.py` | `BioViLTDataset` + `biovilt_collate_fn` — variable-K per-sample, padded to `(B, K_batch, 3, H, W)` + `(B, K_batch)` bool mask. |
| `biovilt/migrate_checkpoint.py` | Upstream BioViL-T `(2,1,256) → (K_max+1, 1, 256)` checkpoint migration (CLI + programmatic API). |
| `biovilt/resume_train.py` | DDP training script. Flags: `--k-max`, `--mode`, `--init-from`, `--resume`. |
| `biovilt/test_smoke.py` | 32 architectural invariants. |
| `biovilt/test_migration.py` | 3 end-to-end migration tests (incl. bit-identical K=1 behavior preservation). |
| `MULTIPRIOR_PLAN.md` | Full design doc — layer-by-layer rationale, file index, what each test enforces. |
| `colab/run_on_colab.ipynb` | One-click Colab demo. |

## Usage

```python
import sys; sys.path.insert(0, 'biovilt')
import torch
from tempcxr.modules.tempcxr_model import TempCXR

model = TempCXR(mode='biovilt', K_max=4).cuda().eval()

B, K = 3, 4
curr_imgs  = torch.randn(B, 3, 448, 448).cuda()
prior_imgs = torch.randn(B, K, 3, 448, 448).cuda()
prior_mask = torch.tensor([                  # variable K per sample:
    [False, False, False, False],            #   sample 0: K_i = 0
    [True,  True,  False, False],            #   sample 1: K_i = 2
    [True,  True,  True,  True ],            #   sample 2: K_i = 4
], dtype=torch.bool).cuda()
texts = ['No acute findings.', 'Stable cardiomegaly.', 'Worsening opacities.']

with torch.no_grad():
    out = model(curr_imgs, prior_imgs, prior_mask, texts=texts)

# out['img_global']  : (B, 128)        — K-invariant
# out['img_patches'] : (B, 196, 128)   — K-invariant
```

## Training

```bash
# Reproduce single-prior BioViL-T training exactly (K_max=1)
python biovilt/resume_train.py --k-max 1

# Multi-prior training, fresh init from upstream BioViL-T weights
python biovilt/resume_train.py \
    --k-max 4 --mode biovilt \
    --init-from /path/to/biovil_t_image_model_proj_size_128.pt

# Resume a K_max=1 run at K_max=4 — checkpoint auto-migrates
python biovilt/resume_train.py --k-max 4 --resume /path/to/epoch_40.pt
```

See `MULTIPRIOR_PLAN.md` for the full design walkthrough.

## Acknowledgements

Builds on top of Microsoft's [BioViL-T](https://github.com/microsoft/hi-ml)
(`hi-ml-multimodal` / `health_multimodal`). Upstream is K=1; this repo
generalizes the temporal pooler to arbitrary K_max.
