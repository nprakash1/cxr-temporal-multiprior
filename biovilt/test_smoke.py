"""
test_smoke.py
=============

Unit / smoke tests for the BioViL-T pipeline in biovilt/.

Sections:
    1. STANDALONE MATH (PE migration)
    2. IMAGE ENCODER (current single-prior API)
    3. FULL MODEL (TempCXR with text encoder)
    4. MULTI-PRIOR (K>=2) via per-prior LOOPING — Approach A
    5. HYPOTHETICAL REFACTORED ENCODER — output shape contract
    6. VARIABLE-K BATCHING — pad + mask
    7. BIOVIL-T ARCHITECTURE — flatten + concatenate + self-attn + slice
       (mirrors Figure 2 of the BioViL-T paper) — Approach B

USAGE
-----
    conda activate cxrtemporal
    cd biovilt
    python test_smoke.py
"""

import os
import sys
import traceback

import torch

# ----------------------------------------------------------------------
# Make local imports work whether run from repo root or biovilt/
# ----------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


# ======================================================================
# CONFIG
# ======================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

B = 2          # batch size
H = W = 448    # image height/width
D_EMB = 128    # joint embedding dim
L = 196        # number of patches per image (14x14 for 448/32)


# ======================================================================
# TINY HARNESS
# ======================================================================
_results = []


def _hdr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _run(test_fn, name):
    try:
        test_fn()
        _results.append((name, "PASS", ""))
        print(f"  [✓ PASS] {name}")
    except AssertionError as e:
        _results.append((name, "FAIL", str(e)))
        print(f"  [✗ FAIL] {name}\n          {e}")
    except Exception as e:
        _results.append((name, "ERROR", f"{type(e).__name__}: {e}"))
        print(f"  [! ERROR] {name}\n          {type(e).__name__}: {e}")
        traceback.print_exc()


# ======================================================================
# HELPERS
# ======================================================================
def make_curr(batch=B):
    return torch.randn(batch, 3, H, W, device=DEVICE)


def make_prev(batch=B):
    return torch.randn(batch, 3, H, W, device=DEVICE)


def make_texts(batch=B):
    pool = [
        "Right lower lobe opacity, similar to prior. Heart size normal.",
        "Cardiomegaly. New left pleural effusion.",
        "No acute cardiopulmonary findings.",
        "Increased perihilar opacification compared to prior study.",
    ]
    return [pool[i % len(pool)] for i in range(batch)]


# ======================================================================
# LAZY MODULE CONSTRUCTION
# ======================================================================
_image_encoder = None
_full_model = None


def get_image_encoder():
    global _image_encoder
    if _image_encoder is None:
        from tempcxr.modules.image_encoder import BioViLTImageEncoder
        _image_encoder = BioViLTImageEncoder(mode="biovil").to(DEVICE).eval()
    return _image_encoder


def get_full_model():
    global _full_model
    if _full_model is None:
        from tempcxr.modules.tempcxr_model import TempCXR
        _full_model = TempCXR(mode="biovil").to(DEVICE)
    return _full_model


# ======================================================================
# TESTS — image encoder (current single-prior API)
# ======================================================================
def test_image_encoder_K0():
    enc = get_image_encoder()
    curr = make_curr()
    with torch.no_grad():
        img_global, img_patches = enc(curr, None)
    assert img_global.shape == (B, D_EMB)
    assert img_patches.shape == (B, L, D_EMB)
    assert torch.isfinite(img_global).all()
    assert torch.isfinite(img_patches).all()


def test_image_encoder_K1():
    enc = get_image_encoder()
    curr = make_curr()
    prev = make_prev()
    with torch.no_grad():
        img_global, img_patches = enc(curr, prev)
    assert img_global.shape == (B, D_EMB)
    assert img_patches.shape == (B, L, D_EMB)
    assert torch.isfinite(img_global).all()
    assert torch.isfinite(img_patches).all()


def test_image_encoder_prev_affects_output():
    """Different priors → different outputs. Must use mode='biovilt' because
    in mode='biovil' the temporal transformer is randomly initialised."""
    from tempcxr.modules.image_encoder import BioViLTImageEncoder
    enc = BioViLTImageEncoder(mode="biovilt").to(DEVICE).eval()
    curr = make_curr()
    prev_a = make_prev()
    prev_b = make_prev()
    with torch.no_grad():
        g_a, _ = enc(curr, prev_a)
        g_b, _ = enc(curr, prev_b)
    diff = (g_a - g_b).abs().mean().item()
    assert diff > 1e-5, f"Different priors produced identical outputs (diff={diff:.2e})"


def test_image_encoder_K0_vs_K1_differ():
    enc = get_image_encoder()
    curr = make_curr()
    prev = make_prev()
    with torch.no_grad():
        g_none, _ = enc(curr, None)
        g_with, _ = enc(curr, prev)
    diff = (g_none - g_with).abs().mean().item()
    assert diff > 1e-5, f"K=0 and K=1 produced identical outputs (diff={diff:.2e})"


def test_image_encoder_batch_sizes():
    enc = get_image_encoder()
    failures = []
    for batch in (1, 2, 4):
        curr = make_curr(batch)
        prev = make_prev(batch)
        with torch.no_grad():
            g, p = enc(curr, prev)
        if g.shape != (batch, D_EMB) or p.shape != (batch, L, D_EMB):
            failures.append(f"B={batch}: g={tuple(g.shape)}, p={tuple(p.shape)}")
    assert not failures, "Batch-size invariance failures:\n  " + "\n  ".join(failures)


def test_image_encoder_normalized():
    enc = get_image_encoder()
    curr = make_curr()
    with torch.no_grad():
        g, p = enc(curr, None)
    g_norms = g.norm(dim=-1)
    p_norms = p.norm(dim=-1)
    assert torch.allclose(g_norms, torch.ones_like(g_norms), atol=1e-4)
    assert torch.allclose(p_norms, torch.ones_like(p_norms), atol=1e-4)


# ======================================================================
# TESTS — full TempCXR model
# ======================================================================
def test_full_model_forward_K0():
    model = get_full_model()
    model.eval()
    curr = make_curr()
    texts = make_texts()
    with torch.no_grad():
        out = model(curr, None, texts=texts)
    expected_keys = {
        "img_global", "img_patches",
        "txt_global", "txt_local", "token_mask",
        "mlm_logits", "mlm_labels",
    }
    assert expected_keys.issubset(out.keys())
    assert out["img_global"].shape == (B, D_EMB)
    assert out["img_patches"].shape == (B, L, D_EMB)
    assert out["txt_global"].shape[0] == B
    assert out["txt_global"].shape[1] == D_EMB


def test_full_model_forward_K1():
    model = get_full_model()
    model.eval()
    curr = make_curr()
    prev = make_prev()
    texts = make_texts()
    with torch.no_grad():
        out = model(curr, prev, texts=texts)
    assert out["img_global"].shape == (B, D_EMB)
    assert out["img_patches"].shape == (B, L, D_EMB)


def test_full_model_backward():
    from losses import (
        global_contrastive_loss,
        local_contrastive_loss,
        mlm_loss,
    )
    model = get_full_model()
    model.train()
    curr = make_curr()
    prev = make_prev()
    texts = make_texts()

    out = model(curr, prev, texts=texts)
    loss_g = global_contrastive_loss(out["img_global"], out["txt_global"])
    loss_l = local_contrastive_loss(out["img_patches"], out["txt_local"], out["token_mask"])
    loss_m = mlm_loss(out["mlm_logits"], out["mlm_labels"])
    loss = loss_g + 0.5 * loss_l + loss_m

    assert torch.isfinite(loss)
    loss.backward()

    total_grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    assert total_grad_norm > 0, "No gradients flowed back"
    model.zero_grad(set_to_none=True)


def test_full_model_output_dtypes():
    model = get_full_model()
    model.eval()
    curr = make_curr()
    texts = make_texts()
    with torch.no_grad():
        out = model(curr, None, texts=texts)
    for k in ("img_global", "img_patches", "txt_global", "txt_local"):
        t = out[k]
        assert t.device.type == DEVICE.type
        assert t.dtype == torch.float32


# ======================================================================
# TESTS — multi-prior (K>=2) via per-prior LOOPING — Approach A
# ======================================================================
def _run_with_K_priors(encoder, curr, multi_prior):
    assert multi_prior.dim() == 5
    B_, K_, C_, H_, W_ = multi_prior.shape
    globals_list, patches_list = [], []
    for k in range(K_):
        with torch.no_grad():
            g_k, p_k = encoder(curr, multi_prior[:, k])
        globals_list.append(g_k)
        patches_list.append(p_k)
    return torch.stack(globals_list, dim=1), torch.stack(patches_list, dim=1)


def _aggregate_multi_prior(globals_stack):
    return torch.nn.functional.normalize(globals_stack.mean(dim=1), dim=-1)


def test_multi_prior_shapes_K2():
    enc = get_image_encoder()
    curr = make_curr()
    multi_prior = torch.randn(B, 2, 3, H, W, device=DEVICE)
    g_stack, p_stack = _run_with_K_priors(enc, curr, multi_prior)
    assert g_stack.shape == (B, 2, D_EMB)
    assert p_stack.shape == (B, 2, L, D_EMB)
    assert torch.isfinite(g_stack).all() and torch.isfinite(p_stack).all()


def test_multi_prior_shapes_K3_K4():
    enc = get_image_encoder()
    curr = make_curr()
    for K in (3, 4):
        multi_prior = torch.randn(B, K, 3, H, W, device=DEVICE)
        g_stack, p_stack = _run_with_K_priors(enc, curr, multi_prior)
        assert g_stack.shape == (B, K, D_EMB)
        assert p_stack.shape == (B, K, L, D_EMB)


def test_multi_prior_each_slot_independent():
    from tempcxr.modules.image_encoder import BioViLTImageEncoder
    enc = BioViLTImageEncoder(mode="biovilt").to(DEVICE).eval()
    K = 3
    curr = make_curr()
    multi_prior = torch.randn(B, K, 3, H, W, device=DEVICE)
    g_stack, _ = _run_with_K_priors(enc, curr, multi_prior)
    pairs = [(0, 1), (0, 2), (1, 2)]
    min_diff = min((g_stack[:, i] - g_stack[:, j]).abs().mean().item() for i, j in pairs)
    assert min_diff > 1e-5, f"Two slots collapsed (min diff = {min_diff:.2e})"


def test_multi_prior_mean_pool_is_normalized():
    enc = get_image_encoder()
    curr = make_curr()
    K = 4
    multi_prior = torch.randn(B, K, 3, H, W, device=DEVICE)
    g_stack, _ = _run_with_K_priors(enc, curr, multi_prior)
    agg = _aggregate_multi_prior(g_stack)
    assert agg.shape == (B, D_EMB)
    norms = agg.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_multi_prior_collate_shape_simulation():
    enc = get_image_encoder()
    curr = make_curr()
    with torch.no_grad():
        g_k0, p_k0 = enc(curr, None)
    assert g_k0.shape == (B, D_EMB)
    assert p_k0.shape == (B, L, D_EMB)
    for K in (1, 2, 4):
        multi_prior = torch.randn(B, K, 3, H, W, device=DEVICE)
        g_stack, p_stack = _run_with_K_priors(enc, curr, multi_prior)
        assert g_stack.shape == (B, K, D_EMB)
        assert p_stack.shape == (B, K, L, D_EMB)


def test_multi_prior_aggregation_differs_from_single_prior():
    from tempcxr.modules.image_encoder import BioViLTImageEncoder
    enc = BioViLTImageEncoder(mode="biovilt").to(DEVICE).eval()
    K = 4
    curr = make_curr()
    multi_prior = torch.randn(B, K, 3, H, W, device=DEVICE)
    g_stack, _ = _run_with_K_priors(enc, curr, multi_prior)
    agg = _aggregate_multi_prior(g_stack)
    diffs = [(agg - g_stack[:, k]).abs().mean().item() for k in range(K)]
    assert min(diffs) > 1e-5, f"Agg collapsed to one slot (min={min(diffs):.2e})"


# ======================================================================
# TESTS — hypothetical refactored multi-prior encoder (output shape contract)
# ======================================================================
class _MockMultiPriorEncoder(torch.nn.Module):
    def __init__(self, single_prior_encoder):
        super().__init__()
        self.enc = single_prior_encoder

    def forward(self, curr_imgs, prior_imgs=None):
        if prior_imgs is None:
            return self.enc(curr_imgs, None)
        assert prior_imgs.dim() == 5
        B_, K_, C_, H_, W_ = prior_imgs.shape
        g_list, p_list = [], []
        for k in range(K_):
            g_k, p_k = self.enc(curr_imgs, prior_imgs[:, k])
            g_list.append(g_k)
            p_list.append(p_k)
        g_stack = torch.stack(g_list, dim=1)
        p_stack = torch.stack(p_list, dim=1)
        g_fused = torch.nn.functional.normalize(g_stack.mean(dim=1), dim=-1)
        p_fused = torch.nn.functional.normalize(p_stack.mean(dim=1), dim=-1)
        return g_fused, p_fused


def _single_prior_ref_shapes(encoder):
    curr = make_curr()
    prev = make_prev()
    with torch.no_grad():
        g, p = encoder(curr, prev)
    return tuple(g.shape), tuple(p.shape)


def test_mp_output_shape_matches_single_prior_at_K1():
    enc = get_image_encoder()
    mp_enc = _MockMultiPriorEncoder(enc)
    sp_g_shape, sp_p_shape = _single_prior_ref_shapes(enc)
    curr = make_curr()
    multi_prior = torch.randn(B, 1, 3, H, W, device=DEVICE)
    with torch.no_grad():
        g, p = mp_enc(curr, multi_prior)
    assert tuple(g.shape) == sp_g_shape
    assert tuple(p.shape) == sp_p_shape


def test_mp_output_shape_invariant_in_K():
    enc = get_image_encoder()
    mp_enc = _MockMultiPriorEncoder(enc)
    sp_g_shape, sp_p_shape = _single_prior_ref_shapes(enc)
    failures = []
    for K in (1, 2, 3, 4):
        curr = make_curr()
        multi_prior = torch.randn(B, K, 3, H, W, device=DEVICE)
        with torch.no_grad():
            g, p = mp_enc(curr, multi_prior)
        if tuple(g.shape) != sp_g_shape or tuple(p.shape) != sp_p_shape:
            failures.append(f"K={K}: g={tuple(g.shape)}, p={tuple(p.shape)}")
    assert not failures, "MP output shape not K-invariant:\n  " + "\n  ".join(failures)


def test_mp_output_shape_K0_falls_back_to_single_prior():
    enc = get_image_encoder()
    mp_enc = _MockMultiPriorEncoder(enc)
    curr = make_curr()
    with torch.no_grad():
        g_sp, p_sp = enc(curr, None)
        g_mp, p_mp = mp_enc(curr, None)
    assert tuple(g_mp.shape) == tuple(g_sp.shape)
    assert tuple(p_mp.shape) == tuple(p_sp.shape)


def test_mp_output_normalized_for_all_K():
    enc = get_image_encoder()
    mp_enc = _MockMultiPriorEncoder(enc)
    failures = []
    for K in (1, 2, 3, 4):
        curr = make_curr()
        multi_prior = torch.randn(B, K, 3, H, W, device=DEVICE)
        with torch.no_grad():
            g, p = mp_enc(curr, multi_prior)
        g_norms = g.norm(dim=-1)
        p_norms = p.norm(dim=-1)
        if not torch.allclose(g_norms, torch.ones_like(g_norms), atol=1e-4):
            failures.append(f"K={K} not normalized")
    assert not failures, "MP not normalized:\n  " + "\n  ".join(failures)


def test_mp_compatible_with_downstream_text_encoder():
    from tempcxr.modules.text_encoder import BioViLTTextEncoder
    enc = get_image_encoder()
    mp_enc = _MockMultiPriorEncoder(enc)
    text_enc = BioViLTTextEncoder(mode="biovil").to(DEVICE).eval()
    curr = make_curr()
    multi_prior = torch.randn(B, 3, 3, H, W, device=DEVICE)
    with torch.no_grad():
        _, img_patches = mp_enc(curr, multi_prior)
    assert img_patches.shape == (B, L, D_EMB)
    texts = make_texts()
    with torch.no_grad():
        mlm_logits, mlm_labels = text_enc.forward_mlm(texts, img_patches)
    assert mlm_logits.shape[0] == B
    assert torch.isfinite(mlm_logits).all()


# ======================================================================
# TESTS — VARIABLE-K BATCHING (per-sample K can differ — pad + mask)
# ======================================================================
class _MockVariableKEncoder(torch.nn.Module):
    def __init__(self, single_prior_encoder):
        super().__init__()
        self.enc = single_prior_encoder

    def forward(self, curr_imgs, prior_imgs=None, prior_mask=None):
        if prior_imgs is None:
            return self.enc(curr_imgs, None)
        B_, K_, C_, H_, W_ = prior_imgs.shape
        if prior_mask is None:
            prior_mask = torch.ones(B_, K_, dtype=torch.bool, device=prior_imgs.device)
        g_list, p_list = [], []
        for k in range(K_):
            g_k, p_k = self.enc(curr_imgs, prior_imgs[:, k])
            g_list.append(g_k)
            p_list.append(p_k)
        g_stack = torch.stack(g_list, dim=1)
        p_stack = torch.stack(p_list, dim=1)
        m = prior_mask.float()
        denom = m.sum(dim=1).clamp(min=1.0)
        g_fused = (g_stack * m.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        p_fused = (p_stack * m.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / \
                  denom.unsqueeze(-1).unsqueeze(-1)
        g_fused = torch.nn.functional.normalize(g_fused, dim=-1)
        p_fused = torch.nn.functional.normalize(p_fused, dim=-1)
        return g_fused, p_fused


def _make_variable_k_batch(per_sample_K, K_max=4):
    B_ = len(per_sample_K)
    prior_imgs = torch.zeros(B_, K_max, 3, H, W, device=DEVICE)
    prior_mask = torch.zeros(B_, K_max, dtype=torch.bool, device=DEVICE)
    for i, k_i in enumerate(per_sample_K):
        if k_i > 0:
            prior_imgs[i, :k_i] = torch.randn(k_i, 3, H, W, device=DEVICE)
            prior_mask[i, :k_i] = True
    return prior_imgs, prior_mask


def test_vark_heterogeneous_batch_shape():
    enc = get_image_encoder()
    vk_enc = _MockVariableKEncoder(enc)
    per_sample_K = [3, 2]
    prior_imgs, prior_mask = _make_variable_k_batch(per_sample_K, K_max=3)
    curr = make_curr(batch=2)
    with torch.no_grad():
        g, p = vk_enc(curr, prior_imgs, prior_mask)
    assert g.shape == (2, D_EMB)
    assert p.shape == (2, L, D_EMB)


def test_vark_mask_actually_ignores_padded_slots():
    enc = get_image_encoder()
    vk_enc = _MockVariableKEncoder(enc)
    per_sample_K = [3, 2]
    prior_imgs, prior_mask = _make_variable_k_batch(per_sample_K, K_max=3)
    curr = make_curr(batch=2)
    with torch.no_grad():
        g_base, _ = vk_enc(curr, prior_imgs, prior_mask)
    prior_imgs_b = prior_imgs.clone()
    prior_imgs_b[1, 2] = torch.randn(3, H, W, device=DEVICE) * 100.0
    with torch.no_grad():
        g_stomp, _ = vk_enc(curr, prior_imgs_b, prior_mask)
    diff_sample0 = (g_base[0] - g_stomp[0]).abs().mean().item()
    diff_sample1 = (g_base[1] - g_stomp[1]).abs().mean().item()
    assert diff_sample0 < 1e-5
    assert diff_sample1 < 1e-5, f"Padded slot leaked (diff={diff_sample1:.2e})"


def test_vark_k_max_independent_of_per_sample_K():
    enc = get_image_encoder()
    vk_enc = _MockVariableKEncoder(enc)
    curr = make_curr(batch=1)
    torch.manual_seed(123)
    real_priors = torch.randn(2, 3, H, W, device=DEVICE)

    def pack(K_max):
        prior_imgs = torch.zeros(1, K_max, 3, H, W, device=DEVICE)
        prior_mask = torch.zeros(1, K_max, dtype=torch.bool, device=DEVICE)
        prior_imgs[0, :2] = real_priors
        prior_mask[0, :2] = True
        return prior_imgs, prior_mask

    p4, m4 = pack(4)
    p8, m8 = pack(8)
    with torch.no_grad():
        g4, _ = vk_enc(curr, p4, m4)
        g8, _ = vk_enc(curr, p8, m8)
    diff = (g4 - g8).abs().mean().item()
    assert diff < 1e-5, f"K_max leaked (diff={diff:.2e})"


def test_vark_K0_sample_in_batch():
    enc = get_image_encoder()
    vk_enc = _MockVariableKEncoder(enc)
    per_sample_K = [0, 3]
    prior_imgs, prior_mask = _make_variable_k_batch(per_sample_K, K_max=3)
    assert prior_mask[0].sum().item() == 0
    assert prior_mask[1].sum().item() == 3
    curr = make_curr(batch=2)
    with torch.no_grad():
        g, p = vk_enc(curr, prior_imgs, prior_mask)
    assert g.shape == (2, D_EMB)
    assert p.shape == (2, L, D_EMB)
    assert torch.isfinite(g).all() and torch.isfinite(p).all()


# ======================================================================
# TESTS — BIOVIL-T ARCHITECTURE: flatten + concat + self-attn + slice
# ======================================================================
#
# Mirrors Figure 2 of the BioViL-T paper:
#
#   P_prior_1..K, P_curr each : (B, L, D)
#                              │
#                              ▼  (+ spatial PE + temporal PE)
#                  [ flatten + concatenate ]
#                              │
#                  H_(0) : (B, (K+1)*L, D)
#                              │
#                              ▼
#               [ Transformer self-attn ]
#                              │
#                  H_out : (B, (K+1)*L, D)
#                              │
#                              ▼
#                  [ slice CURR-L tokens ]
#                              │
#                  P_diff : (B, L, D)
#                              │
#                              ▼
#                  V = P_curr + P_diff       → (B, L, D)
#
# Claims:
#   (i)   H_(0) sequence length = (K+1)*L (was 2L at K=1).
#   (ii)  Self-attention is sequence-length-agnostic — no shape errors at any K.
#   (iii) V output is (B, L, D) for ALL K (incl. K=0).
# ======================================================================

class _BioViLTStyleBlock(torch.nn.Module):
    """Minimal faithful reproduction of the BioViL-T temporal block.

    Add spatial + temporal PE, flatten+concat, run TransformerEncoderLayer,
    slice the curr tokens, residual-add to get V.
    """

    def __init__(self, L_=L, D=64, K_max=4, n_heads=4):
        super().__init__()
        self.L = L_
        self.D = D
        self.K_max = K_max

        self.spatial_pe = torch.nn.Parameter(torch.randn(L_, D) * 0.02)
        # K_max prior rows + 1 curr row
        self.temporal_pe = torch.nn.Parameter(torch.randn(K_max + 1, D) * 0.02)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=4 * D,
            batch_first=True, dropout=0.0,
        )

    def forward(self, P_prior, P_curr, prior_mask=None):
        """
        P_prior    : (B, K, L, D)  or None
        P_curr     : (B, L, D)
        prior_mask : (B, K) bool, True=real, False=padded   (optional)

        Returns: V (B, L, D), and the shape of H_(0) for inspection.
        """
        B_, L_, D_ = P_curr.shape
        Pc = P_curr + self.spatial_pe.unsqueeze(0)

        if P_prior is None:
            # K=0 fast path — self-attn over just curr tokens
            Pc_t = Pc + self.temporal_pe[self.K_max].view(1, 1, D_)
            H_in = Pc_t
            H_out = self.encoder_layer(H_in)
            P_diff = H_out
            V = P_curr + P_diff
            return V, tuple(H_in.shape)

        B_p, K_, L_p, D_p = P_prior.shape
        Pp = P_prior + self.spatial_pe.view(1, 1, L_, D_)
        prior_t = self.temporal_pe[:K_].view(1, K_, 1, D_)
        Pp_t = Pp + prior_t
        curr_t = self.temporal_pe[self.K_max].view(1, 1, D_)
        Pc_t = Pc + curr_t

        # FLATTEN + CONCAT — priors first, curr last
        Pp_flat = Pp_t.reshape(B_, K_ * L_, D_)
        H_in = torch.cat([Pp_flat, Pc_t], dim=1)               # (B, (K+1)*L, D)

        # Build key_padding_mask if requested
        kpm = None
        if prior_mask is not None:
            prior_kpm = ~prior_mask.repeat_interleave(L_, dim=1)   # (B, K*L)
            curr_kpm = torch.zeros(B_, L_, dtype=torch.bool, device=H_in.device)
            kpm = torch.cat([prior_kpm, curr_kpm], dim=1)          # (B, (K+1)*L)

        H_out = self.encoder_layer(H_in, src_key_padding_mask=kpm)

        # SLICE the curr-L tokens (last L)
        P_diff = H_out[:, K_ * L_:, :]                          # (B, L, D)

        V = P_curr + P_diff                                     # (B, L, D)
        return V, tuple(H_in.shape)


def _make_patch_grids(B_, K_, L_=L, D_=64):
    P_curr = torch.randn(B_, L_, D_, device=DEVICE)
    P_prior = torch.randn(B_, K_, L_, D_, device=DEVICE) if K_ > 0 else None
    return P_curr, P_prior


def test_biovilt_arch_seq_length_grows_with_K():
    """Claim (i): H_(0) sequence length is (K+1)*L for any K."""
    D_ = 64
    blk = _BioViLTStyleBlock(L_=L, D=D_, K_max=4).to(DEVICE).eval()
    failures = []
    for K_ in (1, 2, 3, 4):
        P_curr, P_prior = _make_patch_grids(B, K_, L_=L, D_=D_)
        with torch.no_grad():
            _, H_in_shape = blk(P_prior, P_curr)
        expected = (B, (K_ + 1) * L, D_)
        if H_in_shape != expected:
            failures.append(f"K={K_}: H_(0)={H_in_shape}, expected {expected}")
    assert not failures, "H_(0) seq length wrong:\n  " + "\n  ".join(failures)


def test_biovilt_arch_V_shape_invariant_in_K():
    """Claim (iii): V has shape (B, L, D) for ALL K (incl. K=0)."""
    D_ = 64
    blk = _BioViLTStyleBlock(L_=L, D=D_, K_max=4).to(DEVICE).eval()
    failures = []
    P_curr, _ = _make_patch_grids(B, 0, L_=L, D_=D_)
    with torch.no_grad():
        V, _ = blk(None, P_curr)
    if V.shape != (B, L, D_):
        failures.append(f"K=0: V={tuple(V.shape)}")
    for K_ in (1, 2, 3, 4):
        P_curr, P_prior = _make_patch_grids(B, K_, L_=L, D_=D_)
        with torch.no_grad():
            V, _ = blk(P_prior, P_curr)
        if V.shape != (B, L, D_):
            failures.append(f"K={K_}: V={tuple(V.shape)}")
    assert not failures, "V shape not K-invariant:\n  " + "\n  ".join(failures)


def test_biovilt_arch_self_attn_runs_at_every_K():
    """Claim (ii): TransformerEncoderLayer handles (K+1)*L tokens at every K."""
    D_ = 64
    blk = _BioViLTStyleBlock(L_=L, D=D_, K_max=4).to(DEVICE).eval()
    for K_ in (1, 2, 3, 4):
        P_curr, P_prior = _make_patch_grids(B, K_, L_=L, D_=D_)
        with torch.no_grad():
            V, _ = blk(P_prior, P_curr)
        assert torch.isfinite(V).all(), f"K={K_}: V has NaN/Inf"


def test_biovilt_arch_curr_attends_to_all_priors():
    """Perturbing ANY prior changes V → curr tokens attend to every prior."""
    D_ = 64
    blk = _BioViLTStyleBlock(L_=L, D=D_, K_max=4).to(DEVICE).eval()
    K_ = 3
    torch.manual_seed(42)
    P_curr, P_prior = _make_patch_grids(B, K_, L_=L, D_=D_)
    with torch.no_grad():
        V_base, _ = blk(P_prior, P_curr)
    failures = []
    for k_to_perturb in range(K_):
        P_prior_b = P_prior.clone()
        P_prior_b[:, k_to_perturb] = torch.randn(B, L, D_, device=DEVICE)
        with torch.no_grad():
            V_perturbed, _ = blk(P_prior_b, P_curr)
        diff = (V_base - V_perturbed).abs().mean().item()
        if diff < 1e-6:
            failures.append(f"prior_{k_to_perturb}: diff={diff:.2e}")
    assert not failures, "Curr doesn't attend to all priors:\n  " + "\n  ".join(failures)


def test_biovilt_arch_pdiff_residual_shape():
    """P_diff has shape (B, L, D) so V = P_curr + P_diff is well-typed."""
    D_ = 64
    blk = _BioViLTStyleBlock(L_=L, D=D_, K_max=4).to(DEVICE).eval()
    K_ = 4
    P_curr, P_prior = _make_patch_grids(B, K_, L_=L, D_=D_)
    with torch.no_grad():
        V, H_in_shape = blk(P_prior, P_curr)
        P_diff = V - P_curr
    assert H_in_shape == (B, (K_ + 1) * L, D_), \
        f"H_(0) wrong: {H_in_shape}"
    assert P_diff.shape == (B, L, D_), \
        f"P_diff shape {tuple(P_diff.shape)} != (B, L, D)"
    assert V.shape == P_curr.shape


def test_biovilt_arch_key_padding_mask_isolates_padded_priors():
    """key_padding_mask zeros out padded prior slots in the (K+1)*L self-attn."""
    D_ = 64
    blk = _BioViLTStyleBlock(L_=L, D=D_, K_max=4).to(DEVICE).eval()
    K_ = 3
    torch.manual_seed(7)
    P_curr, P_prior = _make_patch_grids(B, K_, L_=L, D_=D_)
    # Sample 0: all 3 priors real. Sample 1: only 2 real (slot 2 padded).
    prior_mask = torch.tensor(
        [[True, True, True],
         [True, True, False]],
        device=DEVICE,
    )
    with torch.no_grad():
        V_base, _ = blk(P_prior, P_curr, prior_mask=prior_mask)
    P_prior_b = P_prior.clone()
    P_prior_b[1, 2] = torch.randn(L, D_, device=DEVICE) * 100.0
    with torch.no_grad():
        V_stomp, _ = blk(P_prior_b, P_curr, prior_mask=prior_mask)
    diff_sample0 = (V_base[0] - V_stomp[0]).abs().mean().item()
    diff_sample1 = (V_base[1] - V_stomp[1]).abs().mean().item()
    assert diff_sample0 < 1e-5, \
        f"Sample 0 changed when only sample 1 edited (diff={diff_sample0:.2e})"
    assert diff_sample1 < 1e-5, (
        f"Padded prior slot leaked into self-attn (diff={diff_sample1:.2e}). "
        f"key_padding_mask is not isolating padded slots."
    )


# ======================================================================
# TEST — basic math sanity (no model needed)
# ======================================================================
def test_pe_migration_math():
    """The (2,D) -> (K+1,D) PE migration: keep curr row, replicate prior."""
    D = D_EMB
    old_pe = torch.randn(2, D)
    K = 4
    new_pe = torch.zeros(K + 1, D)
    new_pe[0] = old_pe[0]
    new_pe[1:] = old_pe[1].unsqueeze(0).expand(K, -1)
    assert new_pe.shape == (K + 1, D)
    assert torch.allclose(new_pe[0], old_pe[0])
    for r in range(1, K + 1):
        assert torch.allclose(new_pe[r], old_pe[1])


# ======================================================================
# MAIN
# ======================================================================
def main():
    print(f"Device: {DEVICE}")
    print(f"B={B}, H={W}, D_EMB={D_EMB}, L={L}\n")

    _hdr("STANDALONE MATH (no model needed)")
    _run(test_pe_migration_math,            "PE migration math (2,D)->(K+1,D)")

    _hdr("IMAGE ENCODER — current single-prior API")
    _run(test_image_encoder_K0,             "Image encoder forward, K=0 (no prior)")
    _run(test_image_encoder_K1,             "Image encoder forward, K=1 (one prior)")
    _run(test_image_encoder_prev_affects_output,
         "Different priors produce different outputs")
    _run(test_image_encoder_K0_vs_K1_differ,
         "K=0 and K=1 produce different outputs (prior has effect)")
    _run(test_image_encoder_batch_sizes,    "Shapes invariant across batch sizes")
    _run(test_image_encoder_normalized,     "Output embeddings are L2-normalized")

    _hdr("FULL MODEL — TempCXR with text encoder")
    _run(test_full_model_forward_K0,        "Full forward, K=0")
    _run(test_full_model_forward_K1,        "Full forward, K=1")
    _run(test_full_model_backward,          "Full forward + 3 losses + backward")
    _run(test_full_model_output_dtypes,     "Output dtypes and devices")

    _hdr("MULTI-PRIOR (K >= 2) via per-prior looping")
    _run(test_multi_prior_shapes_K2,
         "Multi-prior K=2 stacking: (B,K,128) and (B,K,L,128)")
    _run(test_multi_prior_shapes_K3_K4,
         "Multi-prior K=3 and K=4 stacking shapes")
    _run(test_multi_prior_each_slot_independent,
         "Each prior slot produces a distinct embedding (K dim not collapsing)")
    _run(test_multi_prior_mean_pool_is_normalized,
         "Mean-pooled aggregation stays L2-normalized")
    _run(test_multi_prior_collate_shape_simulation,
         "Multi-prior collate-style batch at K in {0,1,2,4}")
    _run(test_multi_prior_aggregation_differs_from_single_prior,
         "Aggregated K=4 embedding differs from any single-prior K=1 embedding")

    _hdr("HYPOTHETICAL REFACTORED MULTI-PRIOR ENCODER — output shape contract")
    _run(test_mp_output_shape_matches_single_prior_at_K1,
         "Refactored encoder at K=1 matches single-prior output shape exactly")
    _run(test_mp_output_shape_invariant_in_K,
         "Refactored encoder output shape invariant for K in {1,2,3,4}")
    _run(test_mp_output_shape_K0_falls_back_to_single_prior,
         "Refactored encoder at K=0 (None) falls back to single-prior shape")
    _run(test_mp_output_normalized_for_all_K,
         "Refactored encoder outputs L2-normalized for every K")
    _run(test_mp_compatible_with_downstream_text_encoder,
         "Refactored encoder output is drop-in compatible with text encoder")

    _hdr("VARIABLE-K BATCHING (per-sample K can differ — pad + mask)")
    _run(test_vark_heterogeneous_batch_shape,
         "Heterogeneous batch [K=3, K=2] still yields (B,128) / (B,L,128)")
    _run(test_vark_mask_actually_ignores_padded_slots,
         "Mask is respected: editing a padded slot does not change output")
    _run(test_vark_k_max_independent_of_per_sample_K,
         "K_max=4 vs K_max=8 give same result for a K=2 sample (mask works)")
    _run(test_vark_K0_sample_in_batch,
         "A K=0 sample can coexist with K>0 samples in the same batch")

    _hdr("BIOVIL-T ARCHITECTURE — flatten + concat + self-attn + slice")
    _run(test_biovilt_arch_seq_length_grows_with_K,
         "H_(0) seq length = (K+1)*L for K in {1,2,3,4}")
    _run(test_biovilt_arch_V_shape_invariant_in_K,
         "V output shape (B,L,D) is K-invariant (incl. K=0)")
    _run(test_biovilt_arch_self_attn_runs_at_every_K,
         "TransformerEncoderLayer handles (K+1)*L tokens at every K")
    _run(test_biovilt_arch_curr_attends_to_all_priors,
         "Curr tokens attend to EVERY prior (perturbing any one changes V)")
    _run(test_biovilt_arch_pdiff_residual_shape,
         "P_diff has shape (B,L,D) so V = P_curr + P_diff is well-typed")
    _run(test_biovilt_arch_key_padding_mask_isolates_padded_priors,
         "key_padding_mask isolates padded prior slots in the (K+1)*L self-attn")

    # Summary
    _hdr("SUMMARY")
    n_pass = sum(1 for _, s, _ in _results if s == "PASS")
    n_fail = sum(1 for _, s, _ in _results if s == "FAIL")
    n_err  = sum(1 for _, s, _ in _results if s == "ERROR")
    print(f"  {n_pass} passed   {n_fail} failed   {n_err} errored\n")
    for name, status, msg in _results:
        icon = {"PASS": "✓", "FAIL": "✗", "ERROR": "!"}[status]
        line = f"  [{icon} {status}] {name}"
        if msg:
            first = msg.splitlines()[0]
            line += f"\n          {first}"
        print(line)

    sys.exit(0 if (n_fail == 0 and n_err == 0) else 1)


if __name__ == "__main__":
    main()
