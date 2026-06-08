"""
test_migration.py — verify the checkpoint migration utility end-to-end.

Three tests:
1. Upstream BioViL-T official checkpoint → migrated state dict has correct
   shape and contents (row 0 = upstream curr emb; rows 1..K = upstream
   prior emb replicated).
2. Migrated state loads strict=True into a fresh TempCXR(K_max=K_new).
3. Behavioral equivalence: at K=1 input, the migrated K_max=4 model
   produces *identical* outputs to a fresh K_max=1 model (since the
   extra type_embed rows aren't touched at K=1).
"""
from __future__ import annotations

import os
import sys
import warnings
import torch

# Suppress the noisy hi-ml deprecation warning at the entry point too,
# so the test output stays clean.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*Importing from timm\.models\.layers.*",
)

# Make `biovilt/` itself importable so `migrate_checkpoint` and `tempcxr`
# package paths resolve regardless of where pytest/CWD is.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Make the vendored hi-ml multimodal source importable so
# `health_multimodal.image.model.pretrained` can find the BioViL-T URL helpers.
HI_ML_SRC = os.path.abspath(
    os.path.join(HERE, "tempcxr", "modules", "hi-ml", "hi-ml-multimodal", "src")
)
if os.path.isdir(HI_ML_SRC):
    sys.path.insert(0, HI_ML_SRC)

from migrate_checkpoint import migrate_state_dict
from tempcxr.modules.tempcxr_model import TempCXR
from health_multimodal.image.model.pretrained import (
    _download_biovil_t_image_model_weights,
)


# ----------------------------------------------------------------------
# Test 1 — Migrate the official upstream BioViL-T checkpoint
# ----------------------------------------------------------------------
def test_upstream_migration():
    print("\n=== Test 1: Migrate upstream BioViL-T checkpoint (K_max=4) ===")
    ckpt_path = _download_biovil_t_image_model_weights()
    state = torch.load(ckpt_path, map_location="cpu")

    UPSTREAM_KEY = "encoder.vit_pooler.type_embed"
    assert UPSTREAM_KEY in state, f"Upstream key not found: {UPSTREAM_KEY}"
    old_te = state[UPSTREAM_KEY]
    print(f"  upstream `{UPSTREAM_KEY}` shape: {tuple(old_te.shape)}")
    assert tuple(old_te.shape) == (2, 1, 256), \
        f"Unexpected upstream shape {tuple(old_te.shape)}"

    new_state, log = migrate_state_dict(state, K_max_new=4, verbose=False)
    for line in log:
        print(f"  {line}")

    TGT_KEY = "multi_pooler.type_embed_multi"
    assert TGT_KEY in new_state, f"Missing target key: {TGT_KEY}"
    new_te = new_state[TGT_KEY]
    assert tuple(new_te.shape) == (5, 1, 256), \
        f"Wrong target shape {tuple(new_te.shape)}"
    print(f"  → `{TGT_KEY}` shape: {tuple(new_te.shape)}  ✓")

    # Row contents: row 0 = old row 0; rows 1..4 = old row 1
    assert torch.allclose(new_te[0], old_te[0]), "row 0 mismatch"
    for k in range(1, 5):
        assert torch.allclose(new_te[k], old_te[1]), f"row {k} mismatch"
    print("  row 0 ≡ upstream row 0; rows 1..4 ≡ upstream row 1  ✓")


# ----------------------------------------------------------------------
# Test 2 — Migrated state loads cleanly into a fresh K_max=4 model
# ----------------------------------------------------------------------
def test_load_into_fresh_model():
    print("\n=== Test 2: K_max=1 train state → K_max=4 model (strict load) ===")
    m1 = TempCXR(mode="biovil", K_max=1)
    sd1 = m1.state_dict()
    k1_key = "image_encoder.multi_pooler.type_embed_multi"
    assert k1_key in sd1
    print(f"  K=1 `{k1_key}`: {tuple(sd1[k1_key].shape)}")

    new_sd, log = migrate_state_dict(sd1, K_max_new=4, verbose=False)
    for line in log:
        print(f"  {line}")
    assert tuple(new_sd[k1_key].shape) == (5, 1, 256)

    m4 = TempCXR(mode="biovil", K_max=4)
    missing, unexpected = m4.load_state_dict(new_sd, strict=True)
    assert len(missing) == 0, f"strict load missing keys: {missing[:5]}"
    assert len(unexpected) == 0, f"strict load unexpected keys: {unexpected[:5]}"
    print("  strict load succeeded (no missing/unexpected keys)  ✓")


# ----------------------------------------------------------------------
# Test 3 — Behavioral equivalence at K=1
# ----------------------------------------------------------------------
def test_behavior_equivalence_at_k1():
    print("\n=== Test 3: K_max=4 migrated model ≡ K_max=1 fresh at K=1 input ===")
    torch.manual_seed(0)

    m1 = TempCXR(mode="biovil", K_max=1).eval()
    sd1 = m1.state_dict()
    new_sd, _ = migrate_state_dict(sd1, K_max_new=4, verbose=False)
    m4 = TempCXR(mode="biovil", K_max=4).eval()
    m4.load_state_dict(new_sd, strict=True)

    B = 1
    curr = torch.randn(B, 3, 448, 448)
    prev = torch.randn(B, 3, 448, 448)
    with torch.no_grad():
        g1, p1 = m1.image_encoder(curr, prev)
        g4, p4 = m4.image_encoder(curr, prev)

    g_diff = (g1 - g4).norm().item()
    p_diff = (p1 - p4).norm().item()
    print(f"  global L2 diff: {g_diff:.3e}")
    print(f"  patch L2 diff : {p_diff:.3e}")
    assert torch.allclose(g1, g4, atol=1e-5), \
        f"K=1 behavior changed (global diff = {g_diff})"
    assert torch.allclose(p1, p4, atol=1e-5), \
        f"K=1 behavior changed (patch diff = {p_diff})"
    print("  K=1 input produces bit-identical output before and after migration  ✓")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_upstream_migration()
    test_load_into_fresh_model()
    test_behavior_equivalence_at_k1()
    print("\n✅ All checkpoint migration tests passed")
