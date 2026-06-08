"""
TempCXR — top-level BioViL-T orchestration module.

The forward signature accepts up to K_max priors per sample with a boolean
mask. It is backward-compatible with the legacy single-prior call —
`model(curr_imgs, prev_imgs_4D, texts)` still works because the
`BioViLTImageEncoder.forward` underneath auto-promotes 4D `prior_imgs`
to `(B, 1, 3, H, W)`.

Forward returns a dict of representations only. Losses are computed
externally for clean separation.
"""

import os
import sys
import torch
import torch.nn as nn
from typing import List, Optional

# ------------------------------------------------------------------
# Make project root visible so ../../losses.py is importable
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Import encoders (single source of truth)
# ---------------------------------------------------------
from .image_encoder import BioViLTImageEncoder
from .text_encoder import BioViLTTextEncoder

# ---------------------------------------------------------
# Import losses from ../../losses.py
# ---------------------------------------------------------
from losses import (
    global_contrastive_loss,
    local_contrastive_loss,
    mlm_loss,
)


# =========================================================
# TEMPCXR MODEL
# =========================================================
class TempCXR(nn.Module):
    """
    BioViL-T forward orchestration module with multi-prior support.

    Constructor
    -----------
    mode             : 'biovil' | 'biovilt' | 'biovilt_finetuned'
    K_max            : maximum priors per sample (default 4)

    Forward
    -------
    forward(
        curr_imgs   : (B, 3, H, W),
        prior_imgs  : (B, K, 3, H, W)  or  (B, 3, H, W)  or  None,
        prior_mask  : (B, K)           or  None,
        texts       : list[str] of len B,
    )
    → dict of representations (see below).
    """

    def __init__(
        self,
        mode: str = "biovil",
        K_max: int = 4,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.image_encoder = BioViLTImageEncoder(
            mode=mode,
            checkpoint_path=checkpoint_path,
            K_max=K_max,
        )
        self.text_encoder = BioViLTTextEncoder(mode=mode)
        self.K_max = K_max

    # --------------------------------------------------
    # FULL FORWARD (NO LOSSES)
    # --------------------------------------------------
    def forward(
        self,
        curr_imgs: torch.Tensor,
        prior_imgs: Optional[torch.Tensor] = None,
        prior_mask: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
    ):
        """
        Returns dict with everything needed for loss computation:
            img_global  : (B, 128)
            img_patches : (B, L, 128)
            txt_global  : (B, 128)
            txt_local   : (B, T, 128)
            token_mask  : (B, T)
            mlm_logits  : (B, T, V)
            mlm_labels  : (B, T)
        """
        if texts is None:
            raise ValueError("`texts` is required (pass as a keyword argument).")

        # -------------------------------
        # Image encoding (multi-prior aware)
        # -------------------------------
        img_global, img_patches = self.image_encoder(
            curr_imgs,
            prior_imgs,
            prior_mask,
        )

        # -------------------------------
        # Text encoding (contrastive)
        # -------------------------------
        txt_global, txt_local, token_mask = (
            self.text_encoder.forward_contrastive(texts)
        )

        # -------------------------------
        # Text encoding (MLM)
        # -------------------------------
        mlm_logits, mlm_labels = (
            self.text_encoder.forward_mlm(texts, img_patches)
        )

        return {
            "img_global": img_global,
            "img_patches": img_patches,
            "txt_global": txt_global,
            "txt_local": txt_local,
            "token_mask": token_mask,
            "mlm_logits": mlm_logits,
            "mlm_labels": mlm_labels,
        }


# =========================================================
# SELF-TEST
# =========================================================
if __name__ == "__main__":
    print("\n🔍 Running TempCXR multi-prior forward + external loss test\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TempCXR(K_max=4).to(device)
    model.eval()

    # --------------------------------------------------
    # Dummy inputs — mixed K batch
    # --------------------------------------------------
    B = 2
    K = 3
    curr_imgs = torch.randn(B, 3, 448, 448).to(device)
    prior_imgs = torch.randn(B, K, 3, 448, 448).to(device)
    prior_mask = torch.tensor([
        [True,  True,  False],   # sample 0: 2 real priors
        [True,  False, False],   # sample 1: 1 real prior
    ], dtype=torch.bool, device=device)

    texts = [
        "Increased right pleural effusion.",
        "Left lower lobe pneumonia is improving.",
    ]

    # --------------------------------------------------
    # 1) Multi-prior forward
    # --------------------------------------------------
    with torch.no_grad():
        out = model(curr_imgs, prior_imgs, prior_mask, texts=texts)

    g_loss = global_contrastive_loss(out["img_global"], out["txt_global"])
    l_loss = local_contrastive_loss(
        out["img_patches"], out["txt_local"], out["token_mask"]
    )
    m_loss = mlm_loss(out["mlm_logits"], out["mlm_labels"])

    print("Image global :", tuple(out["img_global"].shape))
    print("Image patches:", tuple(out["img_patches"].shape))
    print("Text global  :", tuple(out["txt_global"].shape))
    print("Text local   :", tuple(out["txt_local"].shape))
    print("MLM logits   :", tuple(out["mlm_logits"].shape))

    print("\n📉 Losses (multi-prior path)")
    print("Global contrastive:", g_loss.item())
    print("Local contrastive :", l_loss.item())
    print("MLM               :", m_loss.item())

    # --------------------------------------------------
    # 2) K=0 fast path (no priors)
    # --------------------------------------------------
    with torch.no_grad():
        out0 = model(curr_imgs, None, None, texts=texts)
    print("\n[K=0 fast path]    global:", tuple(out0["img_global"].shape),
          "patches:", tuple(out0["img_patches"].shape))

    # --------------------------------------------------
    # 3) Legacy single-prior (4-D prior_imgs) — backward compat
    # --------------------------------------------------
    legacy_prev = torch.randn(B, 3, 448, 448).to(device)
    with torch.no_grad():
        out1 = model(curr_imgs, legacy_prev, None, texts=texts)
    print("[legacy 4D prev]   global:", tuple(out1["img_global"].shape),
          "patches:", tuple(out1["img_patches"].shape))

    print("\n✅ TempCXR multi-prior + external losses wired correctly")
