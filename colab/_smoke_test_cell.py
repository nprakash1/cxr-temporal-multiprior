# 🧪 SMOKE TEST — verify multi-prior architecture correctness
# ============================================================
# This cell runs 4 forward-pass scenarios on SYNTHETIC random data to
# verify that the multi-prior changes work. It does NOT do any training.
# Run time: ~3-10 seconds on GPU, ~30 seconds on CPU.
#
# If anything is broken, this cell will fail with AssertionError BEFORE
# you spend 80 minutes training a bad model.
#
# Tests:
#   Section 4 — Temporal embedding migration (init correctness)
#   Section 5 — Scenario A: K=4 batch (all priors present)
#   Section 6 — Scenario B: K=0 batch (no priors -> fast path)
#   Section 7 — Scenario C: Mixed batch [K=0, K=2, K=4]
#   Section 8 — Scenario D: Padding isolation (the hardest case)

import os, sys, contextlib, io
import torch
import torch.nn.functional as F

# Make sure the repo is on the path
REPO_DIR = "/content/cxr-temporal-multiprior"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "biovilt"))

from biovilt.tempcxr.modules.tempcxr_model import TempCXR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32   # smoke test uses fp32 for exact-equality checks

# Color helpers (works in Colab + terminal)
GREEN  = "\033[92m"
RED    = "\033[91m"
BOLD   = "\033[1m"
END    = "\033[0m"

def section(title):
    bar = "─" * 64
    print(f"\n{bar}\n{BOLD}{title}{END}\n{bar}")

def check(label, condition, detail=""):
    tag = f"{GREEN}✅{END}" if condition else f"{RED}❌{END}"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {label}{suffix}")
    if not condition:
        raise AssertionError(label)

# ============================================================
# SECTION 1: BUILD MODEL + REGISTER FORWARD HOOKS
# ============================================================
print(f"{BOLD}══════════════════════════════════════════════════════════════{END}")
print(f"{BOLD} 🧪 SMOKE TEST  ·  TempCXR multi-prior architecture           {END}")
print(f"{BOLD}══════════════════════════════════════════════════════════════{END}")
print(f"Device : {DEVICE}")
print(f"K_max  : 4  ·  B=3  ·  using synthetic random inputs (NOT real X-rays)")

# Build model
print("\nBuilding model (downloads ~441MB BioViL-T weights if not cached)…")
torch.manual_seed(0)
with contextlib.redirect_stdout(io.StringIO()):  # silence model's own init prints
    model = TempCXR(mode="biovilt", K_max=4).to(DEVICE).to(DTYPE).eval()

# Register forward hooks on key modules
hook_handles = []
hook_log = []  # captured shapes per forward pass

def make_hook(name):
    def hook(module, inputs, output):
        in_shapes = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                in_shapes.append(tuple(x.shape))
        if isinstance(output, torch.Tensor):
            out_shape = tuple(output.shape)
        elif isinstance(output, (tuple, list)) and len(output) and isinstance(output[0], torch.Tensor):
            out_shape = tuple(output[0].shape)
        else:
            out_shape = "non-tensor"
        hook_log.append((name, in_shapes, out_shape))
    return hook

ie = model.image_encoder
hook_handles.append(ie.model.encoder.encoder.register_forward_hook(make_hook("ResNet50_trunk")))
hook_handles.append(ie.model.encoder.backbone_to_vit.register_forward_hook(make_hook("backbone_to_vit (1x1 conv)")))
hook_handles.append(ie.multi_pooler.register_forward_hook(make_hook("MultiPriorTransformerPooler")))

# ============================================================
# SECTION 2-3: Helper functions to build synthetic batches
# ============================================================
def make_inputs(num_priors_per_sample, seed=0):
    """Build curr_imgs, prior_imgs, prior_mask for a batch of B samples
    where sample i has num_priors_per_sample[i] real priors."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = len(num_priors_per_sample)
    K_max_local = max(num_priors_per_sample) if num_priors_per_sample else 0
    curr = torch.randn(B, 3, 448, 448, generator=g).to(DEVICE).to(DTYPE)
    if K_max_local == 0:
        return curr, None, None
    priors = torch.randn(B, K_max_local, 3, 448, 448, generator=g).to(DEVICE).to(DTYPE)
    mask = torch.zeros(B, K_max_local, dtype=torch.bool, device=DEVICE)
    for i, k_i in enumerate(num_priors_per_sample):
        mask[i, :k_i] = True
    return curr, priors, mask

texts = ["fake report A", "fake report B", "fake report C"]

# ============================================================
# SECTION 4: MIGRATION CHECK (no forward, inspect weights)
# ============================================================
section("Section 4: Temporal embedding migration (init correctness)")

te_multi = ie.multi_pooler.type_embed_multi.data
te_upstream = ie.multi_pooler.upstream.type_embed.data

check(
    "type_embed_multi shape = (K_max+1, 1, D) = (5, 1, 256)",
    te_multi.shape == (5, 1, 256),
    detail=f"got {tuple(te_multi.shape)}",
)
check(
    "row 0 ≡ upstream curr row",
    torch.allclose(te_multi[0], te_upstream[0]),
)
for k in range(1, 5):
    check(
        f"row {k} ≡ upstream prior row (copy+replicate)",
        torch.allclose(te_multi[k], te_upstream[1]),
    )
check("Migration init correct", True)

# ============================================================
# SECTION 5: SCENARIO A — K=4 batch (all priors)
# ============================================================
section("Section 5: Scenario A — K=4 batch (all samples have 4 priors)")
hook_log.clear()
curr_A, priors_A, mask_A = make_inputs([4, 4, 4], seed=42)

with torch.no_grad():
    out_A = model(curr_A, priors_A, mask_A, texts=texts)

print(f"  Forward pass hook trace:")
for name, in_sh, out_sh in hook_log:
    print(f"    [{name:38s}]  in={in_sh}  →  out={out_sh}")

check(
    "img_global.shape == (3, 128)",
    out_A["img_global"].shape == (3, 128),
    detail=f"got {tuple(out_A['img_global'].shape)}",
)
check(
    "img_patches.shape == (3, 196, 128)",
    out_A["img_patches"].shape == (3, 196, 128),
    detail=f"got {tuple(out_A['img_patches'].shape)}",
)
norms = out_A["img_global"].norm(dim=-1)
check(
    "||img_global[i]|| ≈ 1 (L2-normalized)",
    torch.allclose(norms, torch.ones_like(norms), atol=1e-4),
    detail=f"norms = {norms.tolist()}",
)
check("Scenario A (K=4) correct", True)

# ============================================================
# SECTION 6: SCENARIO B — K=0 batch (no priors, fast path)
# ============================================================
section("Section 6: Scenario B — K=0 batch (no priors → fast path)")
hook_log.clear()
curr_B, priors_B, mask_B = make_inputs([0, 0, 0], seed=42)

with torch.no_grad():
    out_B = model(curr_B, priors_B, mask_B, texts=texts)

print(f"  Forward pass hook trace:")
for name, in_sh, out_sh in hook_log:
    print(f"    [{name:38s}]  in={in_sh}  →  out={out_sh}")

pooler_fired = any(name == "MultiPriorTransformerPooler" for name, _, _ in hook_log)
check(
    "MultiPriorTransformerPooler SKIPPED (fast path via missing_previous_emb)",
    not pooler_fired,
    detail="pooler should not be invoked when no priors anywhere",
)
check(
    "img_global.shape == (3, 128)",
    out_B["img_global"].shape == (3, 128),
)
check(
    "img_patches.shape == (3, 196, 128)",
    out_B["img_patches"].shape == (3, 196, 128),
)

delta_AB = (out_A["img_global"] - out_B["img_global"]).abs().mean().item()
check(
    "Scenario A (with priors) ≠ Scenario B (no priors)",
    delta_AB > 1e-4,
    detail=f"mean |Δ| = {delta_AB:.4e}",
)
check("Scenario B (K=0 fast path) correct", True)

# ============================================================
# SECTION 7: SCENARIO C — Mixed batch [K=0, K=2, K=4]
# ============================================================
section("Section 7: Scenario C — Mixed batch [K=0, K=2, K=4]")
hook_log.clear()

curr_C, priors_C, mask_C = make_inputs([0, 2, 4], seed=42)

with torch.no_grad():
    out_C = model(curr_C, priors_C, mask_C, texts=texts)

print(f"  Forward pass hook trace:")
for name, in_sh, out_sh in hook_log:
    print(f"    [{name:38s}]  in={in_sh}  →  out={out_sh}")

check(
    "img_global.shape == (3, 128)",
    out_C["img_global"].shape == (3, 128),
)
delta_C0_B0 = (out_C["img_global"][0] - out_B["img_global"][0]).abs().max().item()
check(
    "Mixed batch sample 0 (K=0) ≡ Scenario B sample 0 (K=0)",
    delta_C0_B0 < 1e-4,
    detail=f"max |Δ| = {delta_C0_B0:.2e}",
)
delta_C2_A2 = (out_C["img_global"][2] - out_A["img_global"][2]).abs().max().item()
check(
    "Mixed batch sample 2 (K=4) ≡ Scenario A sample 2 (K=4)",
    delta_C2_A2 < 1e-3,
    detail=f"max |Δ| = {delta_C2_A2:.2e} (small numerical drift is OK)",
)
check("Scenario C (mixed batch) correct", True)

# ============================================================
# SECTION 8: SCENARIO D — Padding isolation stress test
# ============================================================
section("Section 8: Scenario D — Padding isolation stress test")

priors_D = priors_C.clone()
torch.manual_seed(999)
priors_D[1, 2:4] = torch.randn_like(priors_D[1, 2:4]) * 100.0

with torch.no_grad():
    out_D = model(curr_C, priors_D, mask_C, texts=texts)

delta_D1 = (out_D["img_global"][1] - out_C["img_global"][1]).abs().max().item()
check(
    "Sample 1 output bit-identical after perturbing padded slots",
    delta_D1 < 1e-4,
    detail=f"max |Δ| = {delta_D1:.2e}  (should be ~0; padded slots must NOT leak)",
)
delta_D2 = (out_D["img_global"][2] - out_C["img_global"][2]).abs().max().item()
check(
    "Sample 2 output unchanged (its slots not perturbed)",
    delta_D2 < 1e-4,
    detail=f"max |Δ| = {delta_D2:.2e}",
)
check("Scenario D (padding isolation) correct", True)

# ============================================================
# WRAP UP
# ============================================================
for h in hook_handles:
    h.remove()

print(f"\n{BOLD}══════════════════════════════════════════════════════════════{END}")
print(f"{BOLD}{GREEN} ALL CHECKS PASSED ✅  ·  Multi-prior architecture verified  {END}")
print(f"{BOLD}══════════════════════════════════════════════════════════════{END}")
print(f"Safe to proceed with training in the next cell.\n")
