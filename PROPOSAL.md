# Extending BioViL-T to Multi-Prior Chest X-Ray Pretraining

## 1. Background

BioViL-T (Bannur et al., CVPR 2023) is a vision-language pretraining model
for chest X-rays that improves over single-image methods by attending to
**one prior X-ray** alongside the current one. This lets the model learn
representations that capture *change over time* (e.g., "the right
pleural effusion has increased"), which matches how radiologists actually
write reports.

The core architectural piece is a small transformer that performs
self-attention over the concatenated patch tokens of the current and
prior images. A learned **temporal embedding** (one vector for
"current", one vector for "prior") tells the transformer which tokens
came from which timepoint.

## 2. Limitation we want to address

BioViL-T uses **exactly one prior image** per sample. This works for
patients with short hospital stays and acute findings, but is limiting
for patients with:

- **Chronic conditions** (e.g., COPD, heart failure) where the
  trajectory over multiple visits matters more than the last comparison.
- **Slow-progressing disease** (e.g., interstitial lung disease, tumor
  growth) where a 6-month-old comparison is more informative than
  yesterday's.
- **Ambiguous acute changes** where a stable baseline from 2 years ago
  is needed to distinguish chronic from new findings.

In real MIMIC-CXR data, ~40% of patients have 3 or more prior X-rays,
but BioViL-T discards all but the most recent. We hypothesize that
using **more historical context** will produce richer representations
and better downstream performance, particularly for chronic-disease
classification and report generation that references long-term
trajectories.

## 3. Proposed extension

We generalize the BioViL-T temporal block from K=1 prior to **K_max ≥ 1
priors** via a single change: the per-timepoint embedding table grows
from 2 rows to `K_max + 1` rows, and the joint self-attention runs over
`(K_max + 1) × 196` patch tokens instead of `2 × 196`. A boolean
key-padding mask zeros out attention to absent priors so samples with
variable K (0 to K_max) all use the same forward pass.

Everything else stays the same: the CNN backbone, the projection heads,
the text encoder (CXR-BERT), the three pretraining losses (global
contrastive, local contrastive, MLM).

### Why this is safe

The new model is **bit-identical** to BioViL-T at K=1. We initialize
the extended embedding table by copying the upstream `current` vector
into row 0 and the upstream `prior` vector into rows 1..K_max. All other
weights (attention blocks, MLPs, norms) are inherited from the BioViL-T
checkpoint unchanged. This means:

- Plugging the new module into a BioViL-T pipeline at K=1 produces
  numerically identical outputs to the original. Verified by
  `biovilt/test_migration.py`.
- We never have to retrain from random init — we can warm-start from
  upstream weights and only spend gradient updates on learning the
  K > 1 differentiation.

### Why this is novel

To our knowledge, no existing CXR vision-language model handles
arbitrary multi-prior context with masked attention. Prior work either
fixes K=1 (BioViL-T, CXR-Foundation), uses a single-summary
"history vector" (CXR-LLaVA), or treats time as a discrete categorical
("baseline"/"recent"/"current" labels without true temporal pooling).
Our approach is the simplest possible generalization of the BioViL-T
recipe to arbitrary K with proper handling of variable history length.

## 4. Method (one paragraph)

The current image and up to K_max prior images are encoded by a shared
ResNet50, producing patch grids of shape (B, 256, 14, 14) per image.
The grids are flattened to 196 tokens each, concatenated into one
sequence of length (K+1) × 196, with each token receiving a spatial
position embedding (shared across all timepoints) and a temporal
embedding (one of K_max+1 learned vectors, indexed by the token's
timepoint slot). Joint self-attention runs over the full sequence
with a key-padding mask that hides absent priors. The first 196
output tokens (corresponding to the current image) are kept,
concatenated channel-wise with the original current-image features,
and projected into a 128-dim joint vision-language space. Pretraining
uses the three BioViL-T losses unchanged: global contrastive (image vs
report), local contrastive (patch vs token), and masked language
modeling with image-conditioned cross-attention.

## 5. Experimental plan

| Phase | Goal | Setup |
|---|---|---|
| **A. Smoke test** | Verify the K=1 path is bit-identical to BioViL-T | Synthetic batch through both models; assert max abs diff < 1e-5 |
| **B. Subset pretraining** | Show training is stable and losses go down | 500-patient MIMIC subset, K∈{1,2,4}, 50 epochs, plot loss curves |
| **C. Full pretraining** | Train on full MIMIC-CXR train set | ~80k image-text pairs, K∈{1,2,4}, batch 128 on multi-GPU |
| **D. Downstream eval** | Quantify the benefit of K > 1 | Linear-probe + fine-tune on CheXpert (14-class) and RSNA Pneumonia; report change relative to K=1 baseline |
| **E. Long-trajectory cohort** | Confirm K > 1 helps where we expect | Subset to patients with ≥4 priors; measure report-level metrics (CIDEr, BLEU, RadGraph-F1) for report generation |

Hypothesis: each phase from B → E will show monotonic improvement of
K=4 over K=1 on patients with long histories, with no degradation on
patients with K=0 (single-visit), because the missing-prior path is
bit-identical to BioViL-T's K=0 fast path.

## 6. Why this is worth doing

- **Clinically grounded**: matches how radiologists read trajectories,
  not snapshots.
- **Minimal architectural change**: one embedding table grows, one
  attention mask is added. Easy to review, easy to reproduce.
- **Strict generalization of BioViL-T**: K=1 reproduces upstream
  exactly, so we can never be worse than baseline on single-prior cases.
- **Data-efficient**: warm-start from BioViL-T weights, fine-tune only
  the K > 1 differentiation.
- **Compute-tractable**: at K_max=4 the joint sequence is 980 tokens,
  fits on a single A100 at batch 128. No new hardware required.

## 7. Deliverables

1. Code (this repo): drop-in replacement for the BioViL-T temporal
   block, plus migration script, smoke tests, and a Colab training
   notebook.
2. Pretrained checkpoints at K_max ∈ {1, 2, 4}.
3. Downstream evaluation numbers on CheXpert + RSNA + a
   long-trajectory subset.
4. A short writeup with loss curves, attention visualizations
   (showing which prior slots get attended to for different findings),
   and an ablation of K_max vs downstream performance.
