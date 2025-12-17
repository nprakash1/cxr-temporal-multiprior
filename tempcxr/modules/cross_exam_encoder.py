import torch
import torch.nn as nn

DEBUG = False  # Turn on for sanity checking

class CrossExamEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_layers=3, num_heads=12, max_patches=576):
        """
        max_patches = 24×24 or whatever your ViT patch count is.
        embed_dim = 768 because ViT-B/16.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_patches = max_patches

        # ---- Learnable POS embeds (S) ----
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim))

        # ---- Learnable TEMPORAL embeds (Xt for current, Xt-1 for previous) ----
        self.temp_embed_current = nn.Parameter(torch.randn(1, max_patches, embed_dim))
        self.temp_embed_prev    = nn.Parameter(torch.randn(1, max_patches, embed_dim))

        # ---- Learnable CLS token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ---- Transformer encoder (3 layers like paper) ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, P, P_prev):
        """
        P:      (B, L, 768) patch embeddings for CURRENT exam
        P_prev: (B, L, 768) patch embeddings for PRIOR exam
        """

        B, L, D = P.shape
        assert D == self.embed_dim, f"Patch dim mismatch: {D} vs {self.embed_dim}"

        # ---- SANITY CHECK 1 ----
        if DEBUG:
            print("\n[CrossExamEncoder] Input shapes:")
            print("  P:      ", P.shape)
            print("  P_prev: ", P_prev.shape)
            if L > self.max_patches:
                raise ValueError(f"L={L} > max_patches={self.max_patches}")

        # Slice positional + temporal embeddings to L tokens
        pos = self.pos_embed[:, :L, :]                # (1, L, D)
        t_cur = self.temp_embed_current[:, :L, :]     # (1, L, D)
        t_prev = self.temp_embed_prev[:, :L, :]       # (1, L, D)

        # ---- Add S + Xt ----
        P_cur  = P      + pos + t_cur
        P_prv  = P_prev + pos + t_prev

        # ---- SANITY CHECK 2 ----
        if DEBUG:
            print("[CrossExamEncoder] After adding S + X_t:")
            print("  P_cur: ", P_cur.shape)
            print("  P_prv: ", P_prv.shape)

        # Expand CLS token to batch
        cls_tok = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # ---- Concatenate [P, P′, CLS] like TempA-VLP formula ----
        x = torch.cat([P_cur, P_prv, cls_tok], dim=1)

        # ---- SANITY CHECK 3 ----
        if DEBUG:
            print("[CrossExamEncoder] Concatenated sequence x:", x.shape)
            # Expect (B, 2L+1, 768)

        # ---- Transformer encoding ----
        x_out = self.transformer(x)  # (B, 2L+1, 768)

        # Output dynamic CLS rep (last token)
        cls_out = x_out[:, -1, :]     # (B, 768)

        # Fused patch embeddings (exclude last CLS)
        patch_out = x_out[:, :-1, :]  # (B, 2L, 768)

        # ---- SANITY CHECK 4 ----
        if DEBUG:
            print("[CrossExamEncoder] Output:")
            print("  cls_out:   ", cls_out.shape)
            print("  patch_out: ", patch_out.shape)
            assert cls_out.ndim == 2 and cls_out.shape[-1] == D
            assert patch_out.ndim == 3 and patch_out.shape[1] == 2*L

        return cls_out, patch_out


# ---------------- Self-test ----------------
'''
if __name__ == "__main__":
    DEBUG = True
    model = CrossExamEncoder(embed_dim=768)  # 8×8 patches

    B, L, D = 2, 576, 768
    P      = torch.randn(B, L, D)
    P_prev = torch.randn(B, L, D)

    cls, patches = model(P, P_prev)
    print("\nFinal outputs:")
    print("CLS:   ", cls.shape)
    print("PATCH: ", patches.shape)
'''
