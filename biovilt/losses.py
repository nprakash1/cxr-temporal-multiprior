import torch
import torch.nn.functional as F
import torch.nn as nn


# =========================================================
# GLOBAL CONTRASTIVE LOSS (InfoNCE)
# =========================================================
def global_contrastive_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard symmetric InfoNCE loss.

    img_emb : (B, D)
    txt_emb : (B, D)
    """

    logits = img_emb @ txt_emb.T
    logits = logits / temperature

    labels = torch.arange(
        img_emb.size(0),
        device=img_emb.device,
    )

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_i2t + loss_t2i)


# =========================================================
# LOCAL CONTRASTIVE LOSS (BioViL-style)
# =========================================================
def local_contrastive_loss(
    img_patches: torch.Tensor,
    txt_tokens: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Local contrastive loss aligning text tokens to image patches.

    img_patches : (B, N, D)
    txt_tokens  : (B, T, D)
    token_mask  : (B, T)  -- True for valid tokens
    """

    # --------------------------------------------------
    # Normalize again for numerical safety
    # --------------------------------------------------
    img_patches = F.normalize(img_patches, dim=-1)
    txt_tokens = F.normalize(txt_tokens, dim=-1)

    # --------------------------------------------------
    # Similarity: (B, T, N)
    # --------------------------------------------------
    sim = torch.einsum("btd,bnd->btn", txt_tokens, img_patches)
    sim = sim / temperature

    # --------------------------------------------------
    # Mask padding tokens
    # --------------------------------------------------
    token_mask = token_mask.unsqueeze(-1)  # (B, T, 1)
    sim = sim.masked_fill(~token_mask, -1e9)

    # --------------------------------------------------
    # Log-softmax over patches
    # --------------------------------------------------
    log_probs = F.log_softmax(sim, dim=-1)

    # --------------------------------------------------
    # Aggregate loss per token, per sample
    # --------------------------------------------------
    token_loss = -log_probs.sum(dim=-1)              # (B, T)
    token_loss = token_loss * token_mask.squeeze(-1)

    valid_tokens = token_mask.sum(dim=1).clamp(min=1)
    sample_loss = token_loss.sum(dim=1) / valid_tokens.squeeze(-1)

    return sample_loss.mean()


# =========================================================
# MLM LOSS (CROSS ENTROPY)
# =========================================================
def mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Standard MLM loss with ignore_index = -100

    logits : (B, T, vocab)
    labels : (B, T)
    """

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    return loss_fn(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

