import torch
import torch.nn.functional as F

def info_nce(img_emb, txt_emb, logit_scale):
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)

    logits = logit_scale * img_emb @ txt_emb.t()
    labels = torch.arange(len(img_emb), device=logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)

    return  (loss_i + loss_t) / 2

