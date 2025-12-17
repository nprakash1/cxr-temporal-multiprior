import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

DEBUG = True   # <<< ENABLE DEBUG


class TextEncoder(nn.Module):
    def __init__(self, model_name="microsoft/BiomedVLP-CXR-BERT-specialized"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # BioViL exposes projection_dim instead of hidden_size
        self.embed_dim = self.model.config.projection_size

        if DEBUG:
            print(f"[TextEncoder] Loaded model: {model_name}")
            print(f"[TextEncoder] Projection dim: {self.embed_dim}")

    @torch.no_grad()
    def forward(self, texts):
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=112,
            return_tensors="pt"
        ).to(self.model.device)

        # 🔑 CRITICAL: use projected multimodal embeddings
        emb = self.model.get_projected_text_embeddings(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
        )

        # Already normalized, but keep for safety
        emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-12)

        if DEBUG:
            print("[TextEncoder] Output embedding shape:", emb.shape)
            print("[TextEncoder] Norm (should be ~1):",
                  emb.norm(dim=-1).mean().item())

        assert emb.ndim == 2, "Text encoder output must be (B, D)"
        return emb


# ---------------- Self-test ----------------
'''
if __name__ == "__main__":
    model = TextEncoder()
    out = model([
        "The lungs are clear with no focal consolidation.",
        "There is interval worsening of the pleural effusion."
    ])
    print("Final output shape:", out.shape)
'''
