import torch
import torch.nn as nn
import torch.nn.functional as F

class TempCXR(nn.Module):
    def __init__(self, text_encoder, image_encoder, cross_encoder, proj_dim=128):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.cross_encoder = cross_encoder

        # -------- Shared-space projection layers (TempA-VLP Sec. 3.1–3.2) --------
        self.proj_img_static   = nn.Linear(768, proj_dim)
        self.proj_img_dynamic  = nn.Linear(768, proj_dim)

    def forward(self, curr_imgs, prev_imgs, static_texts, dynamic_texts):
        """
        Returns:
            vs : static image CLS (B, 128)
            ts : static text CLS  (B, 128)
            vd : dynamic image CLS (B, 128)
            td : dynamic text CLS  (B, 128)
        """

        # ---- STATIC BRANCH ----
        vs, curr_patches = self.image_encoder(curr_imgs)
        ts = self.text_encoder(static_texts)

        # Project into shared space
        vs = self.proj_img_static(vs)

        # Normalize (required for InfoNCE)
        vs = F.normalize(vs, dim=-1)
        ts = F.normalize(ts, dim=-1)

        # ---- PREVIOUS IMAGE PATCHES ----
        prev_cls, prev_patches = self.image_encoder(prev_imgs)

        # ---- CROSS-EXAM ENCODER ----
        vd_cls, vd_patches = self.cross_encoder(curr_patches, prev_patches)

        # Project dynamic image embedding
        vd = self.proj_img_dynamic(vd_cls)
        vd = F.normalize(vd, dim=-1)

        # ---- TEXT DYNAMIC BRANCH ----
        td = self.text_encoder(dynamic_texts)
        td = F.normalize(td, dim=-1)

        # Return patch embeddings for dynamic grounding
        return vs, ts, vd, td, vd_patches

