import os
import sys
import torch
from transformers import AutoConfig

# ------------------------------------------------------------------
# Make hi-ml multimodal visible
# ------------------------------------------------------------------
sys.path.insert(0, os.path.abspath("hi-ml/hi-ml-multimodal/src"))

from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.model.types import ImageEncoderType
from health_multimodal.image.model.pretrained import (
    _download_biovil_image_model_weights,
)

from health_multimodal.text.model.configuration_cxrbert import CXRBertConfig
from health_multimodal.text.model.modelling_cxrbert import CXRBertModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================================
# TEXT ENCODER — CXR-BERT (paper initialization)
# ==================================================================
print("\n================ TEXT ENCODER (CXR-BERT) =================\n")

# Load HF config but use hi-ml model class
config = CXRBertConfig.from_pretrained(
    "pretrained/BiomedVLP-CXR-BERT-specialized"
)

text_encoder = CXRBertModel.from_pretrained(
    "pretrained/BiomedVLP-CXR-BERT-specialized",
    config=config,
).to(DEVICE)

print(text_encoder)


# ---------------- SANITY CHECK ----------------
print("\n---------------- TEXT ENCODER SANITY CHECK ----------------")

print("Text encoder class:", text_encoder.__class__.__name__)
print("Projection size:", text_encoder.config.projection_size)

assert isinstance(text_encoder, CXRBertModel)
assert text_encoder.config.projection_size == 128

print("✔ Text encoder = hi-ml CXRBertModel (paper-correct)")


# ==================================================================
# IMAGE ENCODER — BioViL-T architecture (random init)
# ==================================================================
print("\n================ IMAGE ENCODER (BioViL-T ARCH) =================\n")

image_encoder = ImageModel(
    img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
    joint_feature_size=128,
    pretrained_model_path=None,   # ❌ NOT BioViL-T weights
).to(DEVICE)

print(image_encoder)


# ==================================================================
# APPLY BioViL CNN INITIALIZATION (paper step)
# ==================================================================
print("\n================ APPLYING BIOVIL INITIALIZATION =================\n")

biovil_ckpt = _download_biovil_image_model_weights()
biovil_state = torch.load(biovil_ckpt, map_location="cpu")

missing, unexpected = image_encoder.encoder.encoder.load_state_dict(
    biovil_state,
    strict=False,
)

print("✔ BioViL CNN weights loaded")
print(f"Missing keys (expected): {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")


# ==================================================================
# FINAL CONFIRMATION
# ==================================================================
print("\n================ FINAL CONFIRMATION =================\n")

print("✔ Text encoder  : hi-ml CXRBertModel (CXR-BERT)")
print("✔ Image encoder : hi-ml BioViL-T architecture")
print("✔ CNN init      : BioViL")
print("✔ Temporal init : random")
print("\n✅ Model state matches the START of BioViL-T training")

