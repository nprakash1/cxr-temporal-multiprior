"""One-off helper: insert the disease-entropy diagnostic cells into
colab/train_on_colab.ipynb right after the sanity-check cell (index 9).

Idempotent: if a cell already contains the marker tag, it is replaced
instead of inserting a duplicate. Safe to re-run.
"""
import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "train_on_colab.ipynb")
MARKER = "# >>> DISEASE-ENTROPY DIAGNOSTIC <<<"

markdown_src = """## 2a. Batch disease-entropy check (data quality · no training)

Read-only diagnostic. Joins the official **CheXpert** label CSV to your
train split so every image gets its 14-disease "fingerprint", then
simulates batches at `BATCH_SIZE` and reports two numbers:

- **`profile_entropy_norm`** (0–1, want **~1.0**) — how diverse the
  disease fingerprints are within a batch.
- **`collision_rate`** (0–1, want **~0**) — fraction of images that
  share an identical fingerprint with another image in the same batch.

Low entropy / high collisions ⇒ many identical fingerprints per batch
(usually lots of `No Finding` normals). Those look the same to the model
but the contrastive loss pushes them apart anyway → false negatives →
the kind of noisy signal behind a drifting global-contrastive val loss.
If the numbers are bad, the fix is a disease-aware batch sampler.

**No edits to training/smoke-test/model cells** — pure measurement.
Run it before training so a bad subset costs seconds, not 80 minutes.
"""

code_src = r'''# >>> DISEASE-ENTROPY DIAGNOSTIC <<<
# Read-only data-quality check. Does NOT train, does NOT touch the model.
# ----------------------------------------------------------------------
import os, math, numpy as np, pandas as pd

# 1) CheXpert labels. Default = the filtered subset file committed to the
#    repo (built locally once via biovilt/make_chexpert_subset.py), so it
#    ships with the git clone — no Drive upload or PhysioNet creds needed.
#    Columns: subject_id, study_id, + 14 disease columns
#    (1=present, 0=absent, -1=uncertain, blank=not mentioned).
CHEXPERT_CSV = 'subset_out/chexpert_labels_subset.csv'   # repo-committed (Option A)
# If you didn't commit it, fall back to a Drive upload of the full sheet:
# CHEXPERT_CSV = '/content/drive/MyDrive/mimic-cxr-2.0.0-chexpert.csv'


# Batch size to simulate. Falls back to 32 if you run this before cell 11.
BATCH_SIZE_DIAG = int(globals().get('BATCH_SIZE', 32))
N_SIM_BATCHES   = 200          # how many random batches to average over
SEED            = 0

assert os.path.exists(CHEXPERT_CSV), (
    f'CheXpert CSV not found at {CHEXPERT_CSV}. Upload it to Drive and fix '
    f'CHEXPERT_CSV above.'
)

# 2) Load both tables and align the join key types (silent-merge guard).
train = pd.read_csv('subset_out/biovilt_pretrain_train_imagelevel.csv')
chex  = pd.read_csv(CHEXPERT_CSV)
train['study_id'] = train['study_id'].astype('int64')
chex['study_id']  = chex['study_id'].astype('int64')

# 3) Identify the 14 disease columns (everything that isn't an id column).
id_cols      = {'subject_id', 'study_id'}
disease_cols = [c for c in chex.columns if c not in id_cols]
print(f'disease columns ({len(disease_cols)}): {disease_cols}')

# 4) Join: every training row gets its study's 14-disease fingerprint.
merged = train.merge(chex[['study_id'] + disease_cols], on='study_id', how='left')
matched = merged[disease_cols].notna().any(axis=1).mean()
print(f'rows: {len(merged)}   labels matched: {matched:5.1%}')
assert matched > 0, 'No study_id matched — check the CSV is the right CheXpert sheet.'

# 5) Build a binary "present" fingerprint per row (present=1, else 0),
#    then a compact string id so identical fingerprints compare equal.
present = (merged[disease_cols].fillna(0.0).values == 1.0).astype(np.int8)
fingerprints = np.array([''.join(map(str, row)) for row in present])

# 6) Simulate random batches and average the two metrics.
def batch_metrics(labels, B):
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / n
    H = -(p * np.log(p)).sum()
    ent_norm = H / math.log(B) if B > 1 else 0.0          # normalize by max possible
    collisions = (counts[counts > 1].sum()) / n           # share-a-fingerprint frac
    return ent_norm, collisions

rng = np.random.default_rng(SEED)
ent_list, col_list = [], []
idx = np.arange(len(fingerprints))
for _ in range(N_SIM_BATCHES):
    pick = rng.choice(idx, size=min(BATCH_SIZE_DIAG, len(idx)), replace=False)
    e, c = batch_metrics(fingerprints[pick], BATCH_SIZE_DIAG)
    ent_list.append(e); col_list.append(c)

profile_entropy_norm = float(np.mean(ent_list))
collision_rate       = float(np.mean(col_list))

# 7) Context: what fraction of the whole subset is the single most common
#    fingerprint (usually all-zeros = "No Finding")?
uniq, cnts = np.unique(fingerprints, return_counts=True)
top_frac = cnts.max() / cnts.sum()

print('\n================ BATCH DISEASE-ENTROPY ================')
print(f'  batch size simulated : {BATCH_SIZE_DIAG}  (x{N_SIM_BATCHES} batches)')
print(f'  profile_entropy_norm : {profile_entropy_norm:5.3f}   (want ~1.0)')
print(f'  collision_rate       : {collision_rate:5.3f}   (want ~0.0)')
print(f'  most-common profile  : {top_frac:5.1%} of all rows')
print('======================================================')
if profile_entropy_norm < 0.85 or collision_rate > 0.15:
    print('  ⚠️  Batches collide on identical fingerprints — likely a chunk of')
    print('      "No Finding" normals. Consider a disease-aware batch sampler')
    print('      before trusting the global-contrastive loss.')
else:
    print('  ✅  Batches look diverse. Global-contrastive signal should be clean.')
'''


def make_md(src):
    lines = src.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def make_code(src):
    lines = src.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


with open(NB_PATH) as f:
    nb = json.load(f)

# Remove any previously-inserted diagnostic cells (idempotent re-run).
kept = []
skip_next_md = False
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if MARKER in src:
        # drop this code cell and the markdown immediately before it
        if kept and kept[-1].get("cell_type") == "markdown" and \
                "Batch disease-entropy check" in "".join(kept[-1].get("source", [])):
            kept.pop()
        continue
    kept.append(c)
nb["cells"] = kept

# Find the sanity-check cell (the one that reads the train CSV + asserts path).
insert_at = None
for i, c in enumerate(nb["cells"]):
    if "First sample image not found" in "".join(c.get("source", [])):
        insert_at = i + 1
        break
if insert_at is None:
    insert_at = 10  # fallback to the known index

nb["cells"][insert_at:insert_at] = [make_md(markdown_src), make_code(code_src)]

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Inserted diagnostic cells at index {insert_at} / {insert_at + 1}.")
print(f"Notebook now has {len(nb['cells'])} cells.")
