import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from tempcxr.modules.tempcxr_model import TempCXR
from tempcxr.modules.image_encoder import ImageEncoder
from tempcxr.modules.text_encoder import TextEncoder
from tempcxr.modules.cross_exam_encoder import CrossExamEncoder


# ============================================================
# LOAD MODEL
# ============================================================
def load_model(ckpt_path, device):
    model = TempCXR(
        text_encoder=TextEncoder(),
        image_encoder=ImageEncoder(),
        cross_encoder=CrossExamEncoder(),
        proj_dim=128,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)

    logit_d = ckpt["logit_scale_dynamic"].to(device)

    model.to(device)
    model.eval()
    return model, logit_d


# ============================================================
# IMAGE TRANSFORM
# ============================================================
transform = T.Compose([
    T.Resize((512, 512)),
    T.CenterCrop((384, 384)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_image(path, device):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)
    return img.to(device)


# ============================================================
# PROMPT TEMPLATE (EXACT FROM PAPER)
# ============================================================
PROMPT_TEMPLATE = "The progression of {} is {}."


# ============================================================
# FIXED 28+ PHRASE PROGRESSION VOCABULARY
# ============================================================
PROGRESSION_PHRASES = {
    "improving": [
        "improving",
        "improved",
        "interval improvement",
        "improved compared to prior",
        "decreased",
        "interval decrease",
        "resolving",
        "partial resolution",
        "significant improvement",
    ],
    "stable": [
        "stable",
        "unchanged",
        "no significant change",
        "no interval change",
        "stable compared with prior",
        "unchanged from prior examination",
        "without interval progression",
        "no evidence of progression",
        "no significant interval progression",
        "persistent but unchanged",
        "remains stable compared to prior",
        "stable appearance compared to prior",
    ],
    "worsening": [
        "worsening",
        "worsened",
        "progressed",
        "progression",
        "interval increase",
        "increased",
        "interval worsening",
        "worsening appearance",
        "new or increased",
        "deteriorated",
    ],
}

CLS_ORDER = ["improving", "stable", "worsening"]


# ============================================================
# ENCODE PROMPTS
# ============================================================
@torch.no_grad()
def encode_prompts(model, disease, device):
    class_embeddings = {}

    for cls, phrases in PROGRESSION_PHRASES.items():
        embs = []

        for phrase in phrases:
            text = PROMPT_TEMPLATE.format(disease, phrase)

            td = model.text_encoder([text])
            td = F.normalize(td, dim=-1)

            embs.append(td)

        class_embeddings[cls] = torch.cat(embs, dim=0).to(device)

    return class_embeddings


# ============================================================
# CLASSIFY ONE SAMPLE
# ============================================================
@torch.no_grad()
def classify_one(model, logit_d, prev_img, curr_img, disease, device):
    _, curr_patches = model.image_encoder(curr_img)
    _, prev_patches = model.image_encoder(prev_img)

    vd_cls, _ = model.cross_encoder(curr_patches, prev_patches)

    vd = model.proj_img_dynamic(vd_cls)
    vd = F.normalize(vd, dim=-1)

    text_embs = encode_prompts(model, disease, device)

    scores = {}
    for cls, embs in text_embs.items():
        sim = logit_d.exp() * (vd @ embs.T)
        scores[cls] = sim.max().item()

    pred = max(scores, key=scores.get)

    return pred, [
        scores["improving"],
        scores["stable"],
        scores["worsening"],
    ]


# ============================================================
# MAIN LOOP
# ============================================================
if __name__ == "__main__":
    device = "cuda:0"
    ckpt = "old_checkpoint/tempa_epoch_45.pt"

    model, logit_d = load_model(ckpt, device)

    input_csv = "/home/eprakash/temporal/baselines/mscxrt_labels.csv"
    output_dir = "preds_tempa_mscxrt_45"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="TempA-VLP Zero-shot"):
        out_path = os.path.join(output_dir, f"{idx}.csv")
        if os.path.exists(out_path):
            continue

        try:
            prev_img = load_image(row["img_path_prev"], device)
            curr_img = load_image(row["img_path_curr"], device)
            disease  = row["disease_name"]

            pred, scores = classify_one(
                model, logit_d,
                prev_img, curr_img,
                disease, device,
            )

        except Exception as e:
            print(e)
            pred = "ERROR"
            scores = ["", "", ""]

        pd.DataFrame([{
            "disease_name": disease,
            "true_comparison": row["comparison"],
            "predicted_comparison": pred,
            "score_improving": scores[0],
            "score_stable": scores[1],
            "score_worsening": scores[2],
        }]).to_csv(out_path, index=False)

    print("✅ DONE — stable collapse fixed")

