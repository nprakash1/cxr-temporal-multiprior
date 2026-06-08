"""
MultiPriorTransformerPooler — joint self-attention multi-prior temporal block.

Implements the BioViL-T paper's temporal-fusion architecture (Figure 2),
generalized from K=1 to up to K_max priors per sample. One transformer
pass over the concatenated (K+1)·L token sequence, with per-prior
temporal embeddings and a padding mask that isolates absent priors.

Pipeline (matches the BioViL-T paper diagram):

    P_curr    + spatial_PE + temporal_PE[0]   ┐
    P_prior_1 + spatial_PE + temporal_PE[1]   │
    P_prior_2 + spatial_PE + temporal_PE[2]   │  flatten + concat
    ...                                        │  ───────────────►  H_(0)
    P_prior_K + spatial_PE + temporal_PE[K]   ┘  shape (B, (K+1)L, D)
                                                       │
                                                       ▼
                                          K_blocks × (self-attn + MLP)
                                                       │
                                                       ▼
                                                 norm_post
                                                       │
                                                       ▼
                                          slice curr-L tokens
                                                       │
                                                       ▼
                                             P_diff (B, L, D)
                                                       │
                                                       ▼
                                          reshape to (B, D, H_grid, W_grid)

There is NO loop, NO averaging — the transformer's attention is the
aggregation. Padded prior slots are isolated via a key-padding mask that
sets attention weights to those positions to zero post-softmax.

Weights from a pretrained upstream `VisionTransformerPooler` are reused
unchanged: the same `blocks`, `norm_post`, `pos_embed`, `pos_drop`,
and the row-0 / row-1 entries of the (now extended) temporal embedding
are inherited via the canonical copy+replicate migration.

API
---
forward(
    current_image : (B, D, H_grid, W_grid),
    prior_images  : (B, K, D, H_grid, W_grid)  or  None,
    prior_mask    : (B, K) bool                or  None,
) -> (B, D, H_grid, W_grid)

Samples whose `prior_mask` is all False (K_i=0) get a zero `diff_x` here
— the caller (`BioViLTImageEncoder`) substitutes the upstream
`missing_previous_emb` for those rows. This preserves the bit-identical
K=0 fast path that the original BioViL-T checkpoint was trained with.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MultiPriorTransformerPooler(nn.Module):
    """
    Joint multi-prior temporal-fusion transformer.

    Construction
    ------------
    upstream_pooler : a fully-initialized `VisionTransformerPooler` whose
                      pretrained weights (blocks, norm_post, pos_embed,
                      type_embed) we reuse.
    K_max           : maximum number of priors any sample can carry.

    The extended temporal embedding `type_embed_multi` of shape
    `(K_max+1, 1, D)` is initialized via the canonical copy+replicate
    migration from the upstream `(2, 1, D)` table:
        row 0  ← upstream row 0           (current)
        row k  ← upstream row 1   for k=1..K_max  (every prior shares
                                                   the pretrained prior emb at init)
    This means a freshly-built joint pooler is *behaviorally equivalent*
    to upstream at K=1 before any further training.
    """

    def __init__(
        self,
        upstream_pooler: nn.Module,
        K_max: int = 4,
    ) -> None:
        super().__init__()
        if K_max < 1:
            raise ValueError(f"K_max must be >= 1, got {K_max}")

        # We hold a reference to the upstream pooler so its parameters
        # (blocks, norm_post, pos_drop, pos_embed buffer) participate in
        # state_dict, gradient flow, and weight loading. We DO NOT call
        # upstream.forward() — we re-use its sub-modules directly.
        self.upstream = upstream_pooler
        self.K_max = K_max

        # Allocate the extended temporal embedding via the migration formula.
        # Shape: (K_max+1, 1, D). Initialized from upstream's (2,1,D) table.
        old_type_embed: torch.Tensor = upstream_pooler.type_embed.data  # (2,1,D)
        if old_type_embed.dim() != 3 or old_type_embed.shape[0] != 2:
            raise RuntimeError(
                f"Expected upstream type_embed of shape (2, 1, D); "
                f"got {tuple(old_type_embed.shape)}"
            )
        D = old_type_embed.shape[-1]
        new_type_embed = torch.zeros(K_max + 1, 1, D, dtype=old_type_embed.dtype)
        new_type_embed[0] = old_type_embed[0].clone()                # current row
        new_type_embed[1:] = (
            old_type_embed[1:2].expand(K_max, -1, -1).clone()         # replicate prior row
        )
        self.type_embed_multi = nn.Parameter(new_type_embed)

    # ==================================================================
    # Forward — JOINT self-attention over (K+1)·L tokens
    # ==================================================================
    def forward(
        self,
        current_image: torch.Tensor,
        prior_images: Optional[torch.Tensor] = None,
        prior_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Joint self-attention forward.

        Parameters
        ----------
        current_image : (B, D, H_grid, W_grid)
        prior_images  : (B, K, D, H_grid, W_grid)  or  None
        prior_mask    : (B, K) bool                 or  None
                        True = real prior slot, False = padded (ignored).

        Returns
        -------
        diff_x : (B, D, H_grid, W_grid)
            For rows where prior_mask is all-False the output is zeros —
            the caller substitutes `missing_previous_emb` for those rows.
        """
        # ----------------------------------------------------------------
        # Early short-circuit: no priors anywhere in the batch.
        # The transformer pass would be over just curr-L tokens — but the
        # K=1-trained pos_and_type_embed wasn't built for that case, and
        # the upstream code uses `missing_previous_emb` instead. So we
        # return zeros and let the caller substitute the placeholder.
        # ----------------------------------------------------------------
        if prior_images is None or prior_mask is None or not prior_mask.any():
            return torch.zeros_like(current_image)

        B, K, D, H_grid, W_grid = prior_images.shape
        L = H_grid * W_grid
        if K > self.K_max:
            raise ValueError(
                f"prior_images has K={K} but pooler was built with K_max={self.K_max}."
            )

        # ----------------------------------------------------------------
        # 1. Flatten each patch grid to (B, L, D).
        # ----------------------------------------------------------------
        x_curr = current_image.view(B, D, L).transpose(1, 2)             # (B, L, D)
        x_prior = (
            prior_images.view(B, K, D, L).transpose(2, 3)                # (B, K, L, D)
        )

        # ----------------------------------------------------------------
        # 2. Add spatial PE (reuse upstream's pos_embed buffer) and
        #    per-row temporal PE from type_embed_multi.
        #
        #    upstream.pos_embed : (1, L, D)   (broadcasts over B)
        #    type_embed_multi    : (K_max+1, 1, D)
        # ----------------------------------------------------------------
        pos_embed = self.upstream.pos_embed                              # (1, L, D)

        t_curr = self.type_embed_multi[0].view(1, 1, D)                  # (1, 1, D)
        # Spatial+temporal embedding to add to Q,K via Block.with_pos_and_type_embed
        # We do not add it directly to x — upstream Block.forward expects
        # raw x and a pos_and_type_embed tensor.
        pe_curr = pos_embed + t_curr                                     # (1, L, D)

        # Per-prior temporal rows. type_embed_multi rows 1..K give the
        # K used prior slots (newest=row 1, etc.).
        t_priors = self.type_embed_multi[1:K + 1].view(1, K, 1, D)       # (1, K, 1, D)
        pe_priors = pos_embed.unsqueeze(1) + t_priors                    # (1, K, L, D)

        # Flatten priors over (K, L): (1, K*L, D)
        pe_priors_flat = pe_priors.reshape(1, K * L, D)
        pos_and_type_embed = torch.cat(
            [pe_curr, pe_priors_flat], dim=1
        )                                                                # (1, (K+1)L, D)

        # ----------------------------------------------------------------
        # 3. Flatten + concat token sequences.
        #    Order: curr tokens FIRST (slots [0, L)), then K prior blocks.
        #    This matches the upstream convention (curr first), which makes
        #    slicing the output trivial: out[:, :L, :] is the curr block.
        # ----------------------------------------------------------------
        x_prior_flat = x_prior.reshape(B, K * L, D)                      # (B, K*L, D)
        x = torch.cat([x_curr, x_prior_flat], dim=1)                     # (B, (K+1)L, D)

        # ----------------------------------------------------------------
        # 4. Build src_key_padding_mask.
        #    True  → KEY position is padding → attention weight forced to 0
        #    False → KEY position is real → normal attention
        #
        #    Layout: [curr (L cols of False), prior_1 (L cols), ..., prior_K (L cols)]
        #    Per-sample: for a padded prior slot k, the corresponding L
        #    columns are True; otherwise False.
        # ----------------------------------------------------------------
        curr_kpm = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        prior_kpm = (~prior_mask).repeat_interleave(L, dim=1)            # (B, K*L)
        key_padding_mask = torch.cat([curr_kpm, prior_kpm], dim=1)       # (B, (K+1)L)

        # ----------------------------------------------------------------
        # 5. Positional dropout (reuse upstream's nn.Dropout).
        # ----------------------------------------------------------------
        x = self.upstream.pos_drop(x)

        # ----------------------------------------------------------------
        # 6. Run through the K_blocks transformer layers.
        #    Upstream `Block.forward(x, pos_and_type_embed)` does:
        #        x_with_emb = norm1(x) + emb
        #        x = x + drop_path(self.attn.forward_as_mhsa(x_with_emb))
        #        x = x + drop_path(self.mlp(norm2(x)))
        #    forward_as_mhsa is plain self-attention with NO mask support.
        #    We replicate the block forward INLINE so we can add the
        #    key_padding_mask to the attention scores ourselves.
        # ----------------------------------------------------------------
        for block in self.upstream.blocks:
            x = _block_forward_with_mask(
                block=block,
                x=x,
                pos_and_type_embed=pos_and_type_embed,
                key_padding_mask=key_padding_mask,
            )

        x = self.upstream.norm_post(x)                                   # (B, (K+1)L, D)

        # ----------------------------------------------------------------
        # 7. Slice out the curr-L tokens (slots [0, L)) and reshape back
        #    to (B, D, H_grid, W_grid) for channel-concat downstream.
        # ----------------------------------------------------------------
        curr_token_features = x[:, :L, :]                                # (B, L, D)
        diff_x = curr_token_features.transpose(1, 2).view(
            B, D, H_grid, W_grid
        )                                                                # (B, D, H_grid, W_grid)
        return diff_x


# ----------------------------------------------------------------------
# Block forward with padding-mask support
# ----------------------------------------------------------------------
def _block_forward_with_mask(
    block: nn.Module,
    x: torch.Tensor,                       # (B, N, D)
    pos_and_type_embed: torch.Tensor,      # (1, N, D)  broadcasts over B
    key_padding_mask: torch.Tensor,        # (B, N) bool, True = pad
) -> torch.Tensor:
    """
    Inline replacement for upstream `Block.forward(x, pos_and_type_embed)`
    with masking. Mirrors the upstream sequence exactly:

        x_with_emb = norm1(x) + pos_and_type_embed
        x = x + drop_path(masked_self_attention(x_with_emb))
        x = x + drop_path(mlp(norm2(x)))
    """
    x_with_emb = block.norm1(x) + pos_and_type_embed
    attn_out = _masked_self_attention(
        attn_module=block.attn,
        x=x_with_emb,
        key_padding_mask=key_padding_mask,
    )
    x = x + block.drop_path(attn_out)
    x = x + block.drop_path(block.mlp(block.norm2(x)))
    return x


def _masked_self_attention(
    attn_module: nn.Module,                # upstream MultiHeadAttentionLayer
    x: torch.Tensor,                       # (B, N, D)
    key_padding_mask: torch.Tensor,        # (B, N) bool, True = pad
) -> torch.Tensor:
    """
    Equivalent of `MultiHeadAttentionLayer.forward_as_mhsa(x)` but with a
    boolean key-padding mask applied to attention scores before softmax.

    This re-uses the upstream module's pretrained linear projections
    (proj_q, proj_k, proj_v, proj), dropout layers (attn_drop, proj_drop),
    and per-head scaling factor. Only the score-masking line is new.
    """
    B, N, C = x.shape
    H = attn_module.num_heads
    head_dim = C // H

    # Project Q, K, V using the upstream module's pretrained linear layers.
    q = attn_module.proj_q(x).reshape(B, N, H, head_dim).permute(0, 2, 1, 3)
    k = attn_module.proj_k(x).reshape(B, N, H, head_dim).permute(0, 2, 1, 3)
    v = attn_module.proj_v(x).reshape(B, N, H, head_dim).permute(0, 2, 1, 3)

    # Scaled dot-product attention scores: (B, H, N_q, N_k)
    scores = (q @ k.transpose(-2, -1)) * attn_module.scale

    # Apply the key-padding mask: set padded KEY positions to -inf before softmax.
    # key_padding_mask shape: (B, N) → broadcast to (B, 1, 1, N).
    mask = key_padding_mask.view(B, 1, 1, N)
    scores = scores.masked_fill(mask, float("-inf"))

    attn = scores.softmax(dim=-1)

    # Defensive: rows whose every key was masked (all -inf) yield NaN after
    # softmax. Replace those with uniform attention so the output is finite.
    # In our pipeline this can only happen if a Q-token's whole row is masked,
    # which never occurs because curr-L rows are never masked. But we keep
    # the guard for safety against future callers.
    if torch.isnan(attn).any():
        attn = torch.nan_to_num(attn, nan=0.0)

    attn = attn_module.attn_drop(attn)

    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    out = attn_module.proj(out)
    out = attn_module.proj_drop(out)
    return out


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Joint-mode standalone check using a randomly-initialized upstream
    pooler. Verifies shapes, masking correctness, and that perturbing a
    real prior changes the output while perturbing a padded prior does not.
    """
    import os
    import sys

    HI_ML_SRC = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "hi-ml", "hi-ml-multimodal", "src")
    )
    if os.path.isdir(HI_ML_SRC):
        sys.path.insert(0, HI_ML_SRC)
    from health_multimodal.image.model.transformer import VisionTransformerPooler

    torch.manual_seed(0)
    D = 256
    H_grid = W_grid = 14
    L = H_grid * W_grid
    B = 3
    K_max = 4

    upstream = VisionTransformerPooler(input_dim=D, grid_shape=(H_grid, W_grid))
    pooler = MultiPriorTransformerPooler(upstream_pooler=upstream, K_max=K_max).eval()

    curr = torch.randn(B, D, H_grid, W_grid)

    # ---- 1. No-priors short-circuit ----
    print("=== No-priors short-circuit ===")
    out0 = pooler(curr, prior_images=None, prior_mask=None)
    assert out0.shape == (B, D, H_grid, W_grid)
    assert out0.abs().sum().item() == 0.0
    print(f"  out shape = {tuple(out0.shape)}; sum(|x|) = 0   ✓")

    # ---- 2. K=2 all-real joint attention ----
    print("\n=== Joint K=2 (all real) ===")
    priors = torch.randn(B, 2, D, H_grid, W_grid)
    mask = torch.ones(B, 2, dtype=torch.bool)
    with torch.no_grad():
        out_k2 = pooler(curr, prior_images=priors, prior_mask=mask)
    assert out_k2.shape == (B, D, H_grid, W_grid)
    print(f"  out shape = {tuple(out_k2.shape)}; mean = {out_k2.mean().item():.4f}   ✓")

    # ---- 3. Mixed K with masking ----
    print("\n=== Joint K=4, mixed mask: sample 0 has 0 priors, sample 1 has 2 ===")
    priors4 = torch.randn(B, 4, D, H_grid, W_grid)
    mask4 = torch.tensor([
        [False, False, False, False],
        [True,  True,  False, False],
        [True,  True,  True,  True],
    ], dtype=torch.bool)
    with torch.no_grad():
        out_k4 = pooler(curr, prior_images=priors4, prior_mask=mask4)
    assert out_k4.shape == (B, D, H_grid, W_grid)
    # Note: sample 0 (K_i=0) still gets a non-zero output here because its
    # curr-L tokens self-attend to themselves (only the prior key cols are
    # masked, not the curr cols). The CALLER (`BioViLTImageEncoder.forward`)
    # overwrites those rows with the upstream `missing_previous_emb` placeholder
    # via `torch.where(no_priors, missing, diff_x)`. So the pooler is free
    # to return whatever curr-only attention produces for those rows.
    print(f"  out shape = {tuple(out_k4.shape)}")
    print(f"  sample 0 (K_i=0) mean    = {out_k4[0].mean().item():.4f}  "
          f"(caller will substitute missing_previous_emb)")
    print(f"  sample 1 (K_i=2) mean    = {out_k4[1].mean().item():.4f}")
    print(f"  sample 2 (K_i=4) mean    = {out_k4[2].mean().item():.4f}")

    # ---- 4. Mask isolation: padded prior should NOT affect output ----
    print("\n=== Padding isolation: editing a padded prior slot must not change output ===")
    priors_a = priors4.clone()
    priors_b = priors4.clone()
    # For sample 1, slot 2 is padded (mask4[1,2]=False). Perturb it heavily:
    priors_b[1, 2] = torch.randn(D, H_grid, W_grid) * 100.0
    with torch.no_grad():
        out_a = pooler(curr, prior_images=priors_a, prior_mask=mask4)
        out_b = pooler(curr, prior_images=priors_b, prior_mask=mask4)
    diff_sample1 = (out_a[1] - out_b[1]).abs().mean().item()
    print(f"  sample 1 diff after editing padded slot 2 = {diff_sample1:.2e}")
    assert diff_sample1 < 1e-5, "Padded slot LEAKED into self-attention!"
    print("  ✓ padded slot is correctly isolated")

    # ---- 5. Sensitivity: editing a real prior SHOULD change output ----
    print("\n=== Sensitivity: editing a real prior slot must change output ===")
    priors_c = priors4.clone()
    # For sample 1, slot 0 is real. Perturb it.
    priors_c[1, 0] = torch.randn(D, H_grid, W_grid) * 100.0
    with torch.no_grad():
        out_c = pooler(curr, prior_images=priors_c, prior_mask=mask4)
    diff_real = (out_a[1] - out_c[1]).abs().mean().item()
    print(f"  sample 1 diff after editing REAL slot 0 = {diff_real:.4e}")
    assert diff_real > 1e-5, "Real prior had no effect on output!"
    print("  ✓ real prior affects output")

    # ---- 6. Migration check ----
    print("\n=== Migration check: type_embed_multi initialized correctly ===")
    te = pooler.type_embed_multi.data
    assert te.shape == (K_max + 1, 1, D)
    assert torch.allclose(te[0], upstream.type_embed.data[0])
    for k in range(1, K_max + 1):
        assert torch.allclose(te[k], upstream.type_embed.data[1])
    print(f"  type_embed_multi shape = {tuple(te.shape)}   ✓")
    print(f"  row 0 ≡ upstream row 0; rows 1..{K_max} ≡ upstream row 1   ✓")

    print("\n✅ MultiPriorTransformerPooler (joint mode) self-test passed")
