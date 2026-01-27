import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import ViTModel

DEBUG = False

class ImageEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-384"):
        super().__init__()

        # Load pre-trained ViT-B/16
        self.vit = ViTModel.from_pretrained(model_name)

        self.hidden_dim = self.vit.config.hidden_size   # 768
        self.patch_dim = self.hidden_dim                # each patch token is 768

    def forward(self, imgs):
        """
        imgs: list of PIL images or a batch tensor (B,3,H,W)
        returns:
            cls_norm: (B, 768) L2-normalized CLS embeddings
            patch_norm: (B, N, 768) L2-normalized patch embeddings
        """
        out = self.vit(pixel_values=imgs)
        last_hidden = out.last_hidden_state  # (B, 1+N, 768)

        cls = last_hidden[:, 0]              # (B, 768)
        patches = last_hidden[:, 1:]         # (B, N, 768)

        if DEBUG:
            print("[ImageEncoder] CLS shape:", cls.shape)
            print("[ImageEncoder] Patch shape:", patches.shape)

        return cls, patches


# --------------- Self-test ---------------
'''
if __name__ == "__main__":
    DEBUG = True
    import PIL.Image as Image
    img = Image.new("RGB", (1024, 1024), color="white")
    enc = ImageEncoder()
    cls, patches = enc([img])
    print("Output:", cls.shape, patches.shape)
'''

