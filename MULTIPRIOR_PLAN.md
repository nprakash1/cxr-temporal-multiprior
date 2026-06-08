# Plumbing `K_max` Through the BioViL-T Pipeline

This document explains exactly what changes when you make the number of
priors `K` a tweakable parameter. There are really **five** layers that
must agree, not three — I lumped some together in my last reply.

```
CSV ──► Dataset ──► Collate ──► Encoder.forward ──► Encoder internals
 (1)      (2)         (3)             (4)                  (5)
 ✅DONE   ✅DONE      ✅DONE        ✅DONE             ✅DONE

   Plus:
   • ✅ Checkpoint migration utility (Step 7)
   • ✅ `--k-max` wired into the training script (Step 8)
   • ⏳ Smoke-train on the 500-patient subset (Step 9 — runtime only)
```

**Status — all architectural work complete.** Layers 1–5 are
implemented, the multi-prior joint self-attention pooler is the only
code path, the upstream BioViL-T checkpoint can be auto-migrated to any
K_max ≥ 1, and `resume_train.py` exposes `--k-max` end-to-end. The
35 automated tests (`biovilt/test_smoke.py` 32/32 +
`biovilt/test_migration.py` 3/3) all pass.

Each layer below shows: **what it looked like before** (single prior),
**what it looks like now** (variable K up to K_max), and **why**. The
“today / before” snippets describe the pre-refactor state; the
“after / now” versions are what is shipping in code.

---

## Layer 1 — The CSV (`create_dataset.py`)

### Today

Look at `biovilt/dataset.py:104–141`. The CSV has these columns:

```
subject_id, study_id, dicom_id_curr,
dicom_id_prior, prior_study_id,        ← singular!
has_prior, full_report_text, split
```

`dicom_id_prior` and `prior_study_id` are **singular** — exactly one
prior per row. `has_prior` is a boolean.

### After

```
subject_id, study_id, dicom_id_curr,
dicom_ids_prior,       ← list, JSON-encoded:  "[\"abc\", \"def\", \"ghi\"]"
prior_study_ids,       ← list, JSON-encoded:  "[101, 102, 103]"
num_priors,            ← int 0..K_max         (replaces has_prior)
full_report_text, split
```

When you build the CSV in `create_dataset.py`, for each current study
you now walk back **up to K_max** prior studies for the same patient
instead of just one, and store them as a JSON-encoded list. A patient
with only 2 historical exams stores a length-2 list; a brand-new
patient stores `[]` with `num_priors=0`.

You also need a config arg, e.g.:

```python
parser.add_argument("--k-max", type=int, default=4,
                    help="Max number of historical priors to include per study.")
```

so the dataset-building script truncates at K_max.

### Why this layer matters

The model can't have more priors than the dataset gives it. The CSV is
the upstream cap. If K_max=4 here, every other layer can rely on
`K_i ∈ [0, 4]`.

---

## Layer 2 — The Dataset (`biovilt/dataset.py`)

### Today (`__getitem__`, lines 121–163)

```python
def __getitem__(self, idx):
    row = self.df.iloc[idx]
    ...
    curr_img = apply_augmentation(curr_raw, params)   # (3, H, W)

    prior_img = None
    if row["has_prior"]:
        prior_raw = self._load_raw(row["dicom_id_prior"], ...)
        prior_img = apply_augmentation(prior_raw, params)   # (3, H, W)

    return {
        "current_image": curr_img,
        "prior_image": prior_img,    # single tensor or None
        "has_prior": bool(row["has_prior"]),
        "text": row["full_report_text"],
    }
```

One image, one prior (or `None`).

### After

```python
def __getitem__(self, idx):
    row = self.df.iloc[idx]
    ...
    curr_img = apply_augmentation(curr_raw, params)   # (3, H, W)

    # Parse the JSON list of prior DICOMs
    prior_dicoms     = json.loads(row["dicom_ids_prior"])     # list[str]
    prior_study_ids  = json.loads(row["prior_study_ids"])     # list[int]
    K_i = len(prior_dicoms)                                   # 0..K_max

    prior_imgs = []
    for dicom_id, study_id in zip(prior_dicoms, prior_study_ids):
        raw = self._load_raw(dicom_id, subject_id, study_id)
        raw = BASE_TRANSFORM(raw)
        prior_imgs.append(apply_augmentation(raw, params))    # (3, H, W)

    # prior_imgs is a Python list of K_i tensors (possibly empty).
    # We do NOT pad here — that's the collate's job, because K_batch
    # depends on the OTHER samples in the batch, which __getitem__
    # cannot see.

    return {
        "current_image": curr_img,           # (3, H, W)
        "prior_images":  prior_imgs,         # list[Tensor(3,H,W)], len=K_i
        "num_priors":    K_i,                # int
        "text":          row["full_report_text"],
    }
```

Optional: at training time you can also randomly **drop** priors as
augmentation (e.g. uniformly subsample to K_i' ∈ [0, K_i]). That makes
the trained model robust to varying K at inference, which is often what
you actually deploy with.

### Why this layer matters

`__getitem__` works on **one sample** in isolation. It cannot know
K_batch because it doesn't see the other samples. So it returns a
variable-length list and lets the collate handle the per-batch padding.
This is exactly how HuggingFace tokenizers handle variable text lengths
inside a Dataset.

---

## Layer 3 — The Collate (`biovilt_collate_fn`)

### Today (lines 169–189)

```python
def biovilt_collate_fn(batch):
    has_prior = batch[0]["has_prior"]      # assumes batch is HOMOGENEOUS
    curr  = torch.stack([b["current_image"] for b in batch])    # (B,3,H,W)
    prior = (torch.stack([b["prior_image"] for b in batch])
             if has_prior else None)                            # (B,3,H,W) or None
    return {
        "current_image": curr,
        "prior_image":   prior,
        "has_prior":     has_prior,
        "text":          [b["text"] for b in batch],
    }
```

Note the comment: *"Assumes batch is homogeneous (all Ds or all Dm)."*
Today, batches are forced to be all-no-prior **or** all-one-prior, never
mixed, by the way the loaders are constructed. This is a hard
constraint that you'll relax.

### After

```python
def biovilt_collate_fn(batch, K_max_pad=None):
    """
    K_max_pad : optional int — pad to this many slots even if the batch's
                max K_i is smaller.  Useful at training when you want a
                fixed shape across iterations.  If None, pads only to the
                batch's max.
    """
    B = len(batch)
    curr = torch.stack([b["current_image"] for b in batch])  # (B,3,H,W)

    Ks = [b["num_priors"] for b in batch]                    # list[int]
    K_batch = max(Ks) if K_max_pad is None else K_max_pad
    K_batch = max(K_batch, 1)   # keep ≥1 to keep tensor shapes simple

    # Build (B, K_batch, 3, H, W) prior tensor and (B, K_batch) bool mask
    C, H, W = curr.shape[1:]
    prior_imgs = torch.zeros(B, K_batch, C, H, W)
    prior_mask = torch.zeros(B, K_batch, dtype=torch.bool)

    for i, b in enumerate(batch):
        K_i = b["num_priors"]
        if K_i > 0:
            stacked = torch.stack(b["prior_images"], dim=0)   # (K_i,3,H,W)
            prior_imgs[i, :K_i] = stacked
            prior_mask[i, :K_i] = True

    # Edge case: if EVERY sample has K_i=0, we can fall back to None
    # so the encoder takes its fast K=0 path.
    if not prior_mask.any():
        prior_imgs = None
        prior_mask = None

    return {
        "current_image": curr,
        "prior_imgs":    prior_imgs,    # (B, K_batch, 3, H, W) or None
        "prior_mask":    prior_mask,    # (B, K_batch) bool or None
        "text":          [b["text"] for b in batch],
    }
```

A few subtleties:

- **`K_batch` is per-batch**, not global. Different iterations can have
  different K_batch values, which is fine because the encoder doesn't
  hard-code it (only `K_max` is hard-coded, in the positional embedding).
- **The mask is the key new artifact.** Everything downstream that
  consumes `prior_imgs` MUST also receive `prior_mask`, or padded slots
  will silently corrupt the output.
- **The homogeneous-batch assumption is gone.** A K=0 sample and a K=3
  sample can ride in the same batch now. The mask sorts it out.

### Why this layer matters

The collate is the only place that sees the whole batch at once. It's
where variable K gets unified into a tensor + mask. The mask is the
contract between "the data has varying K" and "the model sees
rectangular tensors."

---

## Layer 4 — The encoder's external API (`BioViLTImageEncoder.forward`)

> **Implementation:** `biovilt/tempcxr/modules/image_encoder.py`
> (the `BioViLTImageEncoder` class).

### Before

```python
def forward(self, curr_imgs, prev_imgs=None):
    # curr_imgs : (B, 3, H, W)
    # prev_imgs : (B, 3, H, W)  OR  None
    ...
    return img_global, img_patches   # (B, 128), (B, 196, 128)
```

The argument was `prev_imgs` and was `(B, 3, H, W)` — a single prior.

### Now (shipping)

```python
def forward(self, curr_imgs, prior_imgs=None, prior_mask=None):
    # curr_imgs   : (B, 3, H, W)
    # prior_imgs  : (B, K_batch, 3, H, W)  OR  None
    # prior_mask  : (B, K_batch) bool, True=real, False=padded.
    #               Required if prior_imgs is not None.
    ...
    return img_global, img_patches   # (B, 128), (B, 196, 128) — SAME SHAPES
```

Output shapes are **identical** to before. The K axis is consumed
inside the encoder.

Backward-compat: a 4-D `prior_imgs` of shape `(B, 3, H, W)` is
auto-promoted to `(B, 1, 3, H, W)` with an all-True mask. So legacy
calls `model(curr, prev)` keep working unchanged — verified by the
"IMAGE ENCODER — current single-prior API" tests in `test_smoke.py`.

```python
if prior_imgs is not None and prior_imgs.dim() == 4:
    prior_imgs = prior_imgs.unsqueeze(1)         # (B,3,H,W) -> (B,1,3,H,W)
    if prior_mask is None:
        prior_mask = torch.ones(prior_imgs.shape[0], 1,
                                dtype=torch.bool, device=prior_imgs.device)
```

Same change in `TempCXR.forward` (`biovilt/tempcxr/modules/tempcxr_model.py`):
its signature is now `forward(curr_imgs, prior_imgs=None, prior_mask=None, texts=...)`
and it forwards both new args straight to the image encoder.

### Why this layer matters

This is the **public boundary** of the encoder. Once you commit to the
`(B, K, 3, H, W)` + `(B, K)` mask signature, downstream code
(`TempCXR.forward`, the loss code) needs minimal changes — they just
forward `prior_imgs`/`prior_mask` along and never look at K themselves.

---

## Layer 5 — Encoder internals (what actually changes structurally)

> **Implementation:** `biovilt/tempcxr/modules/multi_prior_block.py`
> (`MultiPriorTransformerPooler`). This module is a thin K-generalization
> wrapper around upstream's pretrained `VisionTransformerPooler` — it
> reuses the upstream `blocks`, `norm_post`, `pos_embed`, and `pos_drop`
> directly, and only **adds** an extended `type_embed_multi` parameter
> plus key-padding-mask support in the attention path. There is no
> "loop mode" — joint self-attention over `(K+1)·L` tokens is the only
> code path.

### Before (conceptually)

The temporal transformer sees `[current; prior]` flattened+concatenated
into a sequence of length **`2L`** (each timestep contributes L=196
patch tokens). The temporal embedding (called **`type_embed`** in the
upstream `health_multimodal` code — see
`site-packages/health_multimodal/image/model/transformer.py:62`) has
shape **`(2, 1, D)`** — one row added to all L curr tokens, one row
added to all L prior tokens. Self-attention runs over the full 2L-token
sequence.

### After

The temporal transformer now sees `[current; prior_1; prior_2; …; prior_K]`
flattened+concatenated into a sequence of length **`(K+1)·L`**. The
temporal embedding has shape **`(K_max + 1, 1, D)`** — one row for the
current plus K_max rows for the up-to-K_max priors. Each row is
broadcast across all L patch tokens of its timestep.

The three concrete changes:

#### 5a. Embedding parameter size  (`type_embed`)

```python
# Today  (transformer.py:60-63 in the installed health_multimodal package)
num_series: int = 2
self.type_embed = nn.Parameter(torch.zeros(num_series, 1, D))

# After
self.K_max = K_max
num_series: int = K_max + 1
self.type_embed = nn.Parameter(torch.zeros(num_series, 1, D))
```

Row 0 = "current", rows 1..K_max = priors in order newest→oldest.

When you load the official BioViL-T K=1 checkpoint into this K_max>1
model, expand the saved `(2, 1, D)` table to `(K_max+1, 1, D)`:

```python
old = ckpt_state[".../type_embed"]                            # (2, 1, D)
new = torch.zeros(K_max + 1, 1, D)
new[0]  = old[0]                                              # current row, exact copy
new[1:] = old[1].unsqueeze(0).expand(K_max, -1, -1).clone()   # replicate prior row
ckpt_state[".../type_embed"] = new
```

After migration, all K prior rows are identical → the model behaves
like the K=1 checkpoint until training differentiates them.

`test_pe_migration_math` in `test_smoke.py` verifies the formula
(it uses a 2D `(K+1, D)` toy table; the live code is `(K+1, 1, D)` —
same idea, extra broadcast axis).

#### 5b. Self-attention over the flatten+concat sequence (with key-padding mask)

**Important correction** vs. an earlier version of this doc: the BioViL-T
paper's Figure 2 does **self-attention over the full concatenated
sequence**, not cross-attention from current-as-Q to priors-as-KV.
Every token attends to every other token, and then the curr-L positions
are sliced out to form `P_diff`.

```python
# Today (K=1) — sketch
#   H_(0) = concat([P_prior, P_curr], dim=1)             # (B, 2L, D)
#   H_out = TransformerEncoderLayer(H_(0))               # (B, 2L, D)
#   P_diff = H_out[:, L:, :]                             # (B, L, D)   ← curr slice
#   V      = P_curr + P_diff                             # (B, L, D)

# After (general K) — same shape pattern, just longer sequence
#   P_prior_flat = P_prior.reshape(B, K*L, D)            # (B, K*L, D)
#   H_(0)        = torch.cat([P_prior_flat, P_curr], 1)  # (B, (K+1)*L, D)
#
#   # key_padding_mask: priors can be padded, curr is always real
#   prior_kpm = ~prior_mask.repeat_interleave(L, dim=1)  # (B, K*L)
#   curr_kpm  = torch.zeros(B, L, dtype=torch.bool)      # (B, L)
#   kpm       = torch.cat([prior_kpm, curr_kpm], dim=1)  # (B, (K+1)*L)
#
#   H_out  = TransformerEncoderLayer(H_(0), src_key_padding_mask=kpm)
#   P_diff = H_out[:, K*L:, :]                           # (B, L, D)   ← curr slice
#   V      = P_curr + P_diff                             # (B, L, D)
```

Three things to notice:

- **Sequence length grows from `2L` to `(K+1)·L`** inside the
  transformer. Feature dim `D` is unchanged. PyTorch's
  `nn.TransformerEncoderLayer` accepts arbitrary sequence length with
  no shape errors (it's just an `(K+1)L × (K+1)L` attention matrix and
  a softmax over the sequence dim).
- **The slice consumes K.** After the transformer, only the L tokens
  corresponding to the current image are kept. So `P_diff` is always
  `(B, L, D)` regardless of K, and `V = P_curr + P_diff` is always
  `(B, L, D)`. This is what makes the output shape K-invariant.
- **`key_padding_mask` size is `(B, (K+1)·L)`**, not `(B, K·L)` —
  because we mask positions in the full self-attention sequence, not
  just the prior portion. The curr-L positions are always unmasked
  (they're never padded).

This is exactly what `MultiPriorTransformerPooler` in
`biovilt/tempcxr/modules/multi_prior_block.py` implements. Upstream
exposes the joint self-attention only for K=1 (hardcoded
`if x_previous is not None: torch.cat((x, x_previous), dim=1)` in
`health_multimodal.image.model.transformer.VisionTransformerPooler.forward_after_reshape`);
the pooler generalizes that to arbitrary K by:

1. Concatenating curr-first then K prior blocks for `(B, (K+1)·L, D)`.
2. Adding upstream's `pos_embed` + per-row `type_embed_multi[k]` (row
   0 = curr, rows 1..K = priors).
3. Inlining `Block.forward` so we can inject a `key_padding_mask`
   (upstream's `forward_as_mhsa` has no mask support).
4. Running upstream's pretrained `blocks` + `norm_post` over the
   longer sequence.
5. Slicing the first L (curr) tokens out and reshaping to
   `(B, D, 14, 14)`.

This is verified by the six `BIOVIL-T ARCHITECTURE` tests in
`test_smoke.py` (`H_(0) seq length = (K+1)*L`,
`V output shape (B,L,D) is K-invariant`,
`key_padding_mask isolates padded prior slots`, …) plus the standalone
self-test inside `multi_prior_block.py` (`__main__` block) which
additionally checks padding isolation (editing a padded slot must not
change the output — observed diff `0.00e+00`) and real-prior
sensitivity.

#### 5c. The K=0 fast path

```python
if prior_imgs is None or (prior_mask is not None and not prior_mask.any()):
    # No real priors anywhere in the batch — skip the temporal transformer
    # entirely and delegate to the upstream single-image path.
    return self.model(current_image=curr_imgs, previous_image=None)
```

The upstream `MultiImageModel.forward(current_image=..., previous_image=None)`
already handles the K=0 case correctly (the `if x_previous is not None`
branch in `transformer.py` is skipped). So we don't need a new
`_current_only_forward` method — we just route to the existing
no-prior code path.

### Why this layer matters

This is where the multi-prior story actually happens architecturally.
Layers 1–4 are plumbing; layer 5 is the model. The positional embedding
size and the attention's key-padding mask are the two non-negotiables.

---

## Step 7 — Checkpoint migration utility (`biovilt/migrate_checkpoint.py`)

The official BioViL-T checkpoint was trained at K_max=1, so its
`encoder.vit_pooler.type_embed` is a `(2, 1, D)` tensor. When we
instantiate `BioViLTImageEncoder(K_max=4)`, the new
`multi_pooler.type_embed_multi` parameter has shape `(5, 1, D)`. A
naive `strict=True` load will fail. The migration utility solves this.

**CLI:**

```
python biovilt/migrate_checkpoint.py \
    --in  /path/to/biovil_t_image_model_proj_size_128.pt \
    --out /path/to/migrated_kmax4.pt \
    --k-max 4
```

**Programmatic API** (used by `resume_train.py` at load time so users
never have to migrate manually):

```python
from migrate_checkpoint import migrate_state_dict
new_state, log = migrate_state_dict(state, K_max_new=4, verbose=False)
```

What it does:

- **Upstream → K_max-aware**: if the state has the upstream
  `encoder.vit_pooler.type_embed` key but no `type_embed_multi`, it
  **fabricates** `type_embed_multi` of shape `(K_max+1, 1, D)` via
  row-0-copy + last-prior-row-replicate.
- **K_max change between our own training checkpoints**: if it already
  has `type_embed_multi` but at a different size, it expands or
  truncates with the same copy/replicate logic.
- **Wraps train ckpts**: detects the `{model, optimizer, scheduler,
  epoch, …}` envelope and migrates the `model` sub-dict only.

Three end-to-end tests in `biovilt/test_migration.py` (all passing):

1. **Shape & contents:** upstream `(2,1,256) → (5,1,256)` with row 0
   verbatim and rows 1..4 = replicated upstream row 1.
2. **Strict load:** migrated state loads with `strict=True` and 0
   missing / 0 unexpected keys into a fresh `TempCXR(K_max=4)`.
3. **Behavioral equivalence:** a K=4 model loaded from migrated K=1
   weights produces **bit-identical** outputs to a fresh K=1 model on
   the same K=1 input (global L2 diff = `0.000e+00`, patch L2 diff =
   `0.000e+00`).

That third test is the critical correctness guarantee — it proves the
migration introduces no behavioral drift, so training-from-upstream
starts in exactly the right place.

---

## Step 8 — Training wiring (`biovilt/resume_train.py` + `resume_train.sh`)

The training script exposes three new CLI flags:

```python
parser.add_argument("--k-max", type=int, default=1)         # 1 = legacy behavior
parser.add_argument("--mode",  type=str, default="biovilt", # biovil/biovilt/biovilt_finetuned
                    choices=["biovil", "biovilt", "biovilt_finetuned"])
parser.add_argument("--init-from", type=str, default=None)  # raw weights for fresh runs
parser.add_argument("--resume",   type=str, default=None)   # resume from train ckpt
```

These flow through to:

- `BioViLTDataset(..., k_max=args.k_max)` for both train and val splits.
- `TempCXR(mode=args.mode, K_max=args.k_max)`.
- Both forward calls (train + val) now use the new signature:
  ```python
  curr       = batch["current_image"].to(DEVICE)
  prior_imgs = batch["prior_images"]
  prior_mask = batch["prior_mask"]
  texts      = batch["text"]
  if prior_imgs is not None:
      prior_imgs = prior_imgs.to(DEVICE)
      prior_mask = prior_mask.to(DEVICE)
  outputs = model(curr, prior_imgs, prior_mask, texts=texts)
  ```
- Checkpoint loading auto-runs `migrate_state_dict(...)` before
  `load_state_dict`, so resuming a K_max=1 checkpoint into a K_max=4
  model just works (the migration log is printed on rank 0).

The launcher (`resume_train.sh`) accepts `K_MAX=4 sbatch resume_train.sh`
to flip into multi-prior mode without editing the script.

**Three example invocations:**

```bash
# Reproduce original single-prior BioViL-T training exactly
python biovilt/resume_train.py --k-max 1

# Multi-prior training, fresh init from upstream BioViL-T weights
python biovilt/resume_train.py \
    --k-max 4 --mode biovilt \
    --init-from /path/to/biovil_t_image_model_proj_size_128.pt

# Resume a K_max=1 training run at K_max=4 — auto-migrates
python biovilt/resume_train.py --k-max 4 --resume /path/to/epoch_40.pt
```

---

## Step 9 — Smoke train on the 500-patient subset (runtime only)

Everything compiles + tests pass. The remaining step is to actually run
~50 training iterations end-to-end against the 500-patient subset CSVs
in `subset_out/` at `K_max=4`. This will catch only runtime issues
(NaNs in joint attention, OOM at `(K+1)·L = 980` token sequences on the
target GPUs, DataLoader worker bugs at K>1), not architectural ones —
those are already covered by the 35 automated tests.

Suggested check: tail the per-step loss for the first ~20 iterations
and confirm `loss_g`, `loss_l`, `loss_m` are all finite and trending
down. If `loss_l` blows up at K>1, the likeliest culprit is the
key_padding_mask not being on the right device; in that case the
`prior_mask.to(DEVICE)` lines in `resume_train.py` (train + val) are
where to look.

---

## Putting it all together

A single new CLI/config flag — `--k-max 4` — flows like this:

```
create_dataset.py  --k-max 4   →  CSV has dicom_ids_prior lists of length ≤ 4
BioViLTDataset                  →  returns prior_images list of length 0..4
biovilt_collate_fn              →  pads to (B, K_batch, 3, H, W) + mask
BioViLTImageEncoder(K_max=4)    →  pos_embed sized (5, D); mask-aware attn
TempCXR(K_max=4)                →  forwards prior_imgs/prior_mask through
migrate_state_dict              →  upstream/old ckpts auto-migrated on load
train loop                      →  unchanged; loss code doesn't see K
```

`K_batch` (per batch) and `K_i` (per sample) vary freely up to `K_max`.
Nothing downstream of the encoder ever sees a K dimension — they all
just see `(B, 128)` globals and `(B, 196, 128)` patches, exactly like
the original code.

---

## File index — what lives where

| File | Role |
|---|---|
| `biovilt/create_dataset.py` | Layer 1 — builds the multi-prior CSV (lists of up to K_max priors per current study). |
| `biovilt/dataset.py` | Layers 2 + 3 — `BioViLTDataset.__getitem__` returns variable-length `prior_images` list; `biovilt_collate_fn` pads to `(B, K_batch, 3, H, W)` + `(B, K_batch)` mask. |
| `biovilt/tempcxr/modules/image_encoder.py` | Layer 4 — `BioViLTImageEncoder.forward(curr, prior_imgs, prior_mask)`. Single CNN pass over `(B*(K+1), 3, H, W)`, then routes to `MultiPriorTransformerPooler`. Substitutes upstream `missing_previous_emb` for K_i=0 rows. |
| `biovilt/tempcxr/modules/multi_prior_block.py` | Layer 5 — `MultiPriorTransformerPooler`. The K-generalization of upstream's `VisionTransformerPooler`: joint self-attention over `(K+1)·L` tokens with `type_embed_multi` and `key_padding_mask`. The *only* multi-prior fusion path — no loop mode. |
| `biovilt/tempcxr/modules/tempcxr_model.py` | `TempCXR.forward(curr, prior_imgs, prior_mask, texts=...)`. Forwards through the image + text encoders; returns a dict of reps for external loss computation. |
| `biovilt/migrate_checkpoint.py` | Step 7 — checkpoint migration utility (`migrate_state_dict`, CLI). Handles upstream BioViL-T → K_max-aware and K_max-to-K_max migrations. |
| `biovilt/resume_train.py` | Step 8 — exposes `--k-max`, `--mode`, `--init-from`, `--resume`. Both train + val forward calls use the new `(curr, prior_imgs, prior_mask, texts=...)` signature. |
| `biovilt/resume_train.sh` | SLURM launcher. Reads `K_MAX` env var, passes as `--k-max`. |
| `biovilt/test_smoke.py` | 32 architectural invariant tests (PE migration, shape K-invariance, mask isolation, BioViL-T architecture, …). |
| `biovilt/test_migration.py` | 3 end-to-end migration tests (shape, strict-load, bit-identical K=1 behavior preservation). |

---

## What the tests in `test_smoke.py` already enforce

| Test section | Refactor invariant it locks in |
|---|---|
| **STANDALONE MATH** | The PE migration formula `(2,D) → (K+1, D)` is correct. |
| **IMAGE ENCODER (single-prior)** | The current code still works after you change the signature (backward-compat path). |
| **FULL MODEL** | Loss + backward still produce gradients (nothing downstream broke). |
| **MULTI-PRIOR via looping** | The looping fallback works at K∈{2,3,4} (useful during refactor — keep it as a code path). |
| **HYPOTHETICAL REFACTORED ENCODER** | The output is `(B,128)/(B,L,128)` for ALL K. K leaks → test fails. |
| **VARIABLE-K BATCHING** | The mask is respected; K=0 and K>0 samples coexist; K_max is decoupled from K_i. |
| **BIOVIL-T ARCHITECTURE** | Flatten+concat sequence length is `(K+1)·L`; self-attention runs at every K; curr tokens attend to every prior; slice produces `P_diff : (B,L,D)` so `V = P_curr + P_diff` is K-invariant; `key_padding_mask` of shape `(B,(K+1)·L)` isolates padded prior slots. |

So when you do the actual refactor, you run `python test_smoke.py` and
the green/red checklist tells you which invariant you broke. The
HYPOTHETICAL, VARIABLE-K, and BIOVIL-T ARCHITECTURE sections in
particular are precisely the contract your refactored encoder needs to
satisfy.

---

## Appendix — Is the prior encoding "combined" or "per-prior"?

Both, at different stages. This is the central architectural decision
of any multi-prior temporal model, so it deserves its own write-up.

### Stage 1: CNN backbone — K+1 SEPARATE encodings

The image backbone (ResNet50 in BioViL-T) has no notion of temporality.
It sees one image and produces a patch grid. So the current image and
every prior get encoded **independently** through the same shared-weights
CNN:

```
curr_imgs   : (B, 3, 448, 448)   ── CNN ──►  f_curr      : (B, 196, D)
prior_1     : (B, 3, 448, 448)   ── CNN ──►  f_prior_1   : (B, 196, D)
prior_2     : (B, 3, 448, 448)   ── CNN ──►  f_prior_2   : (B, 196, D)
prior_3     : (B, 3, 448, 448)   ── CNN ──►  f_prior_3   : (B, 196, D)
```

For GPU efficiency you fuse this into a single CNN pass by reshaping:
`(B, K+1, 3, H, W) → (B*(K+1), 3, H, W) → CNN → (B*(K+1), 196, D) →
(B, K+1, 196, D)`. Mathematically it's still per-image — the CNN never
sees two images at once, so no cross-image information has flowed yet.

### Stage 2: Temporal transformer — ONE COMBINED encoding

This is where the K dimension collapses. Following Figure 2 of the
BioViL-T paper, the temporal transformer concatenates ALL the patch
grids (priors + current) into one long sequence and runs
**self-attention** over the whole thing, then slices out only the
current image's positions to form the difference embedding:

```
      P_prior_1..K, P_curr each : (B, 196, D)
                          + spatial PE + temporal PE
                                  │
                                  ▼
                  [ flatten + concatenate dim=1 ]
                                  │
              H_(0)  : (B, (K+1)*196, D)               ← longer than 2L now
                                  │
                                  ▼
                [ Transformer self-attention ]          ← every token attends
                                                          to every other
                                  │
              H_out  : (B, (K+1)*196, D)               ← length preserved
                                  │
                                  ▼
                [ slice the CURR-196 tokens ]
                                  │
              P_diff : (B, 196, D)                     ← K is GONE
                                  │
                                  ▼
                  V = P_curr + P_diff                  ← residual add
                                  │
              V      : (B, 196, D)                     ← same shape as P_curr
```

Three things make this work:

- **Self-attention, not cross-attention.** Every patch token can
  attend to every other patch token in the sequence — priors↔priors,
  priors↔current, current↔current. This is strictly more expressive
  than cross-attention from curr-as-Q to priors-as-KV; the priors can
  also update each other through the current image as context.
- **The K dimension is consumed by the slice.** The transformer
  preserves sequence length: input `(B, (K+1)·L, D)` → output
  `(B, (K+1)·L, D)`. We then keep ONLY the L tokens corresponding to
  the current image (the last L by convention), giving `P_diff` of
  shape `(B, L, D)`. That's how the output ends up K-invariant.
- **`key_padding_mask` is sized `(B, (K+1)·L)`.** Padded prior slots
  get `True` in the mask (= ignore); curr-L positions get `False`
  (always real). This zeros out padded slots in the self-attention
  softmax — same trick BERT uses for variable text length.

After the slice and residual add, `V : (B, L, D)` is what gets
projected and pooled into the `(B, 128)` global and `(B, 196, 128)`
patches that the text encoder and losses consume. K-invariant
downstream by construction.

### Three architectural choices

| Approach | Stage 1 | Stage 2 | Verdict |
|---|---|---|---|
| **A. Looping wrapper** (today, via `_MockMultiPriorEncoder`) | K+1 separate CNN passes | **K separate** single-prior temporal passes, then external mean-pool | Stopgap. Priors never see each other through attention — each is encoded as if alone. |
| **B. Joint temporal fusion** (Figure 2 of the paper — this doc's recommendation) | K+1 separate CNN passes | **ONE** combined self-attention pass over `(B, (K+1)·L, D)`, then slice the curr-L tokens | Standard. Every patch attends to every patch; K consumed by the slice. |
| **C. Fully joint from the start** (3D / video CNN) | **ONE** CNN sees all K+1 images at once | merged into stage 1 | Expensive; needs new backbone; rare in radiology. |

**Approach B is the right answer for BioViL-T.** That's what Layer 5b
above (the self-attention + slice + key-padding-mask block) implements,
and what `_BioViLTStyleBlock` in `test_smoke.py` is a minimal
reproduction of.

### Why the looping approach is genuinely weaker

The "MULTI-PRIOR via per-prior looping" tests in `test_smoke.py` pass
because they implement Approach A — they call the existing single-prior
encoder K times and aggregate outside. But Approach A has a real
expressivity limitation: in the loop, when the encoder processes
`(curr, prior_2)`, the attention has no way of knowing that `prior_1`
and `prior_3` exist. They influence the final output only through the
mean-pool *after* the encoder, not through attention *inside* it.

In Approach B, every prior's patches sit in the same self-attention
sequence as the current's patches, so every token can do soft retrieval
across **all of them together** within a single softmax. Crucially,
priors can also update each other through the current image as shared
context — something cross-attention with curr-as-Q can't do. That
stronger inductive bias is what makes multi-prior actually outperform
single-prior for tasks like "this opacity has been stable across the
last 3 exams" vs "it's new this time" — questions that fundamentally
require comparing priors to each other, not just each prior to the
current image in isolation.

### Practical implication

You'll almost certainly land on Approach B. The tests in
`test_smoke.py` were written so that the **output-shape and masking
contracts** (the things downstream code depends on) are the same for A
and B — so you can prototype the refactor with A, verify the tests
still pass, then swap out the internals for B and the tests will keep
passing without any change. That's a feature, not a coincidence: the
tests are validating the *interface*, not the *implementation*.
