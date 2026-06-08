# image_encoder.py
"""
BioViLTImageEncoder — multi-prior generalization of the BioViL-T image encoder.

The forward signature now accepts up to K_max priors per sample with a
boolean mask, but it remains backward-compatible with the original
single-prior `(curr, prev)` call: a 4-D `prior_imgs` of shape
`(B, 3, H, W)` is auto-promoted to `(B, 1, 3, H, W)` with an all-True
mask before being fed into the multi-prior path.

Internally:
    if no priors anywhere in the batch:
        delegate to upstream MultiImageModel with previous_image=None
        (this preserves the K=0 fast path bit-for-bit).
    else:
        - run the ResNet50 backbone on torch.cat([curr, prior_1, ..., prior_K], dim=0)
        - run the (1x1 conv) backbone_to_vit projection
        - split into curr-patches and per-prior-patches
        - run MultiPriorTransformerPooler to get diff_x : (B, D, 14, 14)
        - for samples with no real priors (K_i = 0), substitute the
          upstream learned `missing_previous_emb` placeholder
        - re-enter upstream's `forward_post_encoder` to get projected
          patch / global embeddings exactly as before.

The output tensors (`img_global : (B, 128)`, `patch_emb : (B, 196, 128)`)
are shape-identical to the original encoder, so all downstream code
(`TempCXR`, loss functions, train loops) is unaffected.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Silence the noisy upstream FutureWarning emitted by health_multimodal's
# `from timm.models.layers import ...` (deprecated alias in newer timm).
# Upstream still uses the old path; we have no clean way to monkey-patch
# it without forking the library. Filtering only this one warning is the
# minimally-invasive fix.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*Importing from timm\.models\.layers.*",
)
# Also silence transformers' `clean_up_tokenization_spaces` FutureWarning
# (originates from the BERT tokenizer used by the text encoder).
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*clean_up_tokenization_spaces.*",
)

# ------------------------------------------------------------------
# Make hi-ml multimodal visible (vendored mirror, if present)
# ------------------------------------------------------------------
HI_ML_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "hi-ml",
        "hi-ml-multimodal",
        "src",
    )
)
if os.path.isdir(HI_ML_SRC):
    sys.path.insert(0, HI_ML_SRC)

from health_multimodal.image.model.model import MultiImageModel
from health_multimodal.image.model.types import ImageEncoderType
from health_multimodal.image.model.pretrained import (
    _download_biovil_image_model_weights,
    _download_biovil_t_image_model_weights,
)

from .multi_prior_block import MultiPriorTransformerPooler

DEBUG = False


class BioViLTImageEncoder(nn.Module):
    """
    Image encoder supporting three init modes (biovil / biovilt /
    biovilt_finetuned) AND multi-prior temporal context up to K_max priors.

    Construction
    ------------
    mode               : 'biovil' | 'biovilt' | 'biovilt_finetuned'
    checkpoint_path    : required for 'biovilt_finetuned'
    K_max              : max number of priors per sample (default 4)

    Forward
    -------
    forward(
        curr_imgs   : (B, 3, H, W),
        prior_imgs  : (B, K, 3, H, W)  or  (B, 3, H, W)  or  None,
        prior_mask  : (B, K)           or  None,
    ) -> (img_global : (B, 128), patch_emb : (B, L, 128))

    Backward compatibility:
        forward(curr, prev)  with prev shape (B,3,H,W) still works; it is
        auto-promoted to (B,1,3,H,W) with an all-True mask.
    """

    def __init__(
        self,
        mode: str = "biovil",
        checkpoint_path: Optional[str] = None,
        K_max: int = 4,
    ):
        super().__init__()
        assert mode in {"biovil", "biovilt", "biovilt_finetuned"}
        if K_max < 1:
            raise ValueError(f"K_max must be >= 1, got {K_max}")

        self.mode = mode
        self.K_max = K_max
        self.embed_dim = 128

        # --------------------------------------------------
        # Always build the SAME architecture
        # --------------------------------------------------
        self.model = MultiImageModel(
            img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
            joint_feature_size=128,
            pretrained_model_path=None,  # weights loaded manually below
        )

        # --------------------------------------------------
        # Load weights
        # --------------------------------------------------
        if mode == "biovil":
            ckpt = _download_biovil_image_model_weights()
            state = torch.load(ckpt, map_location="cpu")
            # CNN ONLY
            self.model.encoder.encoder.load_state_dict(state, strict=False)
            if DEBUG:
                print("[ImageEncoder] Mode = BioViL (CNN init only)")

        elif mode == "biovilt":
            ckpt = _download_biovil_t_image_model_weights()
            state = torch.load(ckpt, map_location="cpu")
            self.model.load_state_dict(state, strict=True)
            if DEBUG:
                print("[ImageEncoder] Mode = BioViL-T (official pretrained)")

        else:  # biovilt_finetuned
            assert checkpoint_path is not None, \
                "checkpoint_path required for biovilt_finetuned"
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state, strict=True)
            if DEBUG:
                print(f"[ImageEncoder] Mode = BioViL-T (finetuned): {checkpoint_path}")

        # --------------------------------------------------
        # Wrap the upstream vit_pooler with our multi-prior pooler.
        # The upstream pooler's weights are reused as-is; we add an
        # extended type_embed_multi of shape (K_max+1, 1, D) on the side,
        # initialized via the canonical copy+replicate migration.
        # --------------------------------------------------
        upstream_pooler = self.model.encoder.vit_pooler
        self.multi_pooler = MultiPriorTransformerPooler(
            upstream_pooler=upstream_pooler,
            K_max=K_max,
        )

    # ==================================================================
    # Forward
    # ==================================================================
    def forward(
        self,
        curr_imgs: torch.Tensor,
        prior_imgs: Optional[torch.Tensor] = None,
        prior_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        See class docstring.
        """
        # --------------------------------------------------------------
        # Backward-compat: 4D prior_imgs => single prior
        # --------------------------------------------------------------
        if prior_imgs is not None and prior_imgs.dim() == 4:
            prior_imgs = prior_imgs.unsqueeze(1)                # (B, 1, 3, H, W)
            if prior_mask is None:
                prior_mask = torch.ones(
                    prior_imgs.shape[0], 1,
                    dtype=torch.bool,
                    device=prior_imgs.device,
                )

        # --------------------------------------------------------------
        # K=0 fast path — no real priors anywhere in the batch
        # --------------------------------------------------------------
        if prior_imgs is None or prior_mask is None or not prior_mask.any():
            out = self.model(current_image=curr_imgs, previous_image=None)
            return self._postprocess(out)

        # --------------------------------------------------------------
        # Multi-prior path  (K_batch >= 1, at least one real prior in batch)
        # --------------------------------------------------------------
        B, K, C, H, W = prior_imgs.shape
        if curr_imgs.shape != (B, C, H, W):
            raise ValueError(
                f"curr_imgs shape {tuple(curr_imgs.shape)} does not match "
                f"prior_imgs spatial dims {(B, C, H, W)}"
            )
        if prior_mask.shape != (B, K):
            raise ValueError(
                f"prior_mask shape {tuple(prior_mask.shape)} does not match "
                f"prior_imgs (B, K) = {(B, K)}"
            )

        # 1. Stack current + all priors along batch dim → single CNN pass.
        priors_flat = prior_imgs.reshape(B * K, C, H, W)
        all_imgs = torch.cat([curr_imgs, priors_flat], dim=0)        # (B*(K+1), 3, H, W)

        # 2. ResNet50 backbone + 1x1 conv (backbone_to_vit) — same path
        #    the upstream MultiImageEncoder.forward uses internally.
        backbone = self.model.encoder.encoder                        # the resnet50 trunk
        backbone_to_vit = self.model.encoder.backbone_to_vit         # 1x1 conv to D=256

        x_resnet = backbone(all_imgs)                                # (B*(K+1), C_res, 14, 14)
        x = backbone_to_vit(x_resnet)                                # (B*(K+1), 256, 14, 14)

        # 3. Split back into curr-patches and per-prior-patches.
        D, H_g, W_g = x.shape[1], x.shape[2], x.shape[3]
        curr_patches = x[:B]                                         # (B, D, 14, 14)
        prior_patches = x[B:].reshape(B, K, D, H_g, W_g)             # (B, K, D, 14, 14)

        # 4. Multi-prior temporal pooler → (B, D, 14, 14).
        diff_x = self.multi_pooler(
            current_image=curr_patches,
            prior_images=prior_patches,
            prior_mask=prior_mask,
        )

        # 5. Per-sample: rows with K_i=0 get the upstream missing_previous_emb
        #    (a learned placeholder that the K=1 BioViL-T checkpoint already trained).
        no_priors = ~prior_mask.any(dim=1)                           # (B,) bool
        if no_priors.any():
            missing = self.model.encoder.missing_previous_emb        # (1, D, 1, 1)
            missing_bdhw = missing.expand(B, D, H_g, W_g)            # (B, D, 14, 14)
            diff_x = torch.where(
                no_priors.view(B, 1, 1, 1),
                missing_bdhw,
                diff_x,
            )

        # 6. Channel-concat [f_static ; f_diff] → (B, 2D, 14, 14).
        patch_fused = torch.cat([curr_patches, diff_x], dim=1)       # (B, 512, 14, 14)
        avg_pooled = F.adaptive_avg_pool2d(patch_fused, (1, 1))
        avg_pooled = torch.flatten(avg_pooled, 1)                    # (B, 512)

        # 7. Re-enter upstream's projector + global mean-pool exactly as before.
        out = self.model.forward_post_encoder(patch_fused, avg_pooled)
        return self._postprocess(out)

    # ------------------------------------------------------------------
    # Output normalization (identical to the previous encoder)
    # ------------------------------------------------------------------
    def _postprocess(self, out) -> Tuple[torch.Tensor, torch.Tensor]:
        # global embedding: (B, 128)
        img_emb = F.normalize(out.projected_global_embedding, dim=-1)

        # patch embeddings: (B, 128, H', W') → (B, L, 128)
        feat = out.projected_patch_embeddings
        B_, C_, H_, W_ = feat.shape
        patch_emb = F.normalize(
            feat.flatten(2).transpose(1, 2), dim=-1
        )

        if DEBUG:
            print(
                f"[Image:{self.mode}] global:",
                img_emb.shape,
                "mean norm:",
                img_emb.norm(dim=-1).mean().item(),
            )
            print(f"[Image:{self.mode}] patches:", patch_emb.shape)

        return img_emb, patch_emb


# ------------------------------------------------------------------
# SELF-TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🔍 Running BioViLTImageEncoder multi-prior sanity checks\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    K = 3
    curr_imgs = torch.randn(B, 3, 448, 448).to(device)
    prior_imgs = torch.randn(B, K, 3, 448, 448).to(device)
    prior_mask = torch.tensor([
        [True,  True,  False],   # sample 0: 2 real priors
        [False, False, False],   # sample 1: 0 real priors (K=0 fallback per row)
    ], dtype=torch.bool, device=device)

    # ---- BioViL (CNN init only) ----
    print("--- BioViL (CNN init only), K_max=4 ---")
    enc = BioViLTImageEncoder(mode="biovil", K_max=4).to(device)
    enc.eval()

    with torch.no_grad():
        # 1) Legacy single-prior call (auto-promote path)
        legacy_prev = torch.randn(B, 3, 448, 448).to(device)
        g1, p1 = enc(curr_imgs, legacy_prev)
        assert g1.shape == (B, 128)
        assert p1.shape == (B, 196, 128)
        print(f"  legacy (B,3,H,W) prior  → global {tuple(g1.shape)}, patches {tuple(p1.shape)} ✓")

        # 2) New multi-prior call (mixed K mask)
        g2, p2 = enc(curr_imgs, prior_imgs, prior_mask)
        assert g2.shape == (B, 128)
        assert p2.shape == (B, 196, 128)
        print(f"  multi-prior  K=3 mixed  → global {tuple(g2.shape)}, patches {tuple(p2.shape)} ✓")

        # 3) K=0 fast path (no priors at all)
        g3, p3 = enc(curr_imgs, None)
        assert g3.shape == (B, 128)
        assert p3.shape == (B, 196, 128)
        print(f"  K=0 fast path           → global {tuple(g3.shape)}, patches {tuple(p3.shape)} ✓")

    print("\n✅ BioViLTImageEncoder multi-prior sanity checks passed")
