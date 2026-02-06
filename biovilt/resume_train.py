# resume_train.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import (
    BioViLTDataset,
    BioViLTMixedBatchSampler,
    biovilt_collate_fn,
)
from tempcxr.modules.tempcxr_model import TempCXR
from losses import (
    global_contrastive_loss,
    local_contrastive_loss,
    image_text_mlm_loss,
)


# ============================================================
# DDP SETUP
# ============================================================
def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, torch.device(f"cuda:{local_rank}")


local_rank, DEVICE = setup_ddp()
WORLD_SIZE = dist.get_world_size()


# ============================================================
# HYPERPARAMETERS (FROM PAPER)
# ============================================================
LR = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 30          # per GPU
EPOCHS = 50
WARMUP_RATIO = 0.03

# Loss weights (paper)
W_GLOBAL = 1.0
W_LOCAL = 0.5
W_MLM = 1.0


# ============================================================
# DATA
# ============================================================
dataset = BioViLTDataset(
    csv_path="biovilt_pretrain_train_imagelevel.csv",
    image_root="/scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0/files",
    split="train",
    train=True,
)

sampler = BioViLTMixedBatchSampler(
    dataset,
    batch_size=BATCH_SIZE,
    seed=local_rank,
)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=biovilt_collate_fn,
)


# ============================================================
# MODEL
# ============================================================
model = TempCXR().to(DEVICE)
model = DDP(model, device_ids=[local_rank])

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

num_steps = len(loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * num_steps),
    num_training_steps=num_steps,
)


# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(1, EPOCHS + 1):
    model.train()

    if local_rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
    else:
        pbar = loader

    for batch in pbar:
        curr = batch["current_image"].to(DEVICE)
        prior = batch["prior_image"]
        text = batch["text"]

        if prior is not None:
            prior = prior.to(DEVICE)

        outputs = model(
            current_image=curr,
            prior_image=prior,
            text=text,
        )

        loss_global = global_contrastive_loss(outputs)
        loss_local = local_contrastive_loss(outputs)
        loss_mlm = image_text_mlm_loss(outputs)

        loss = (
            W_GLOBAL * loss_global
            + W_LOCAL * loss_local
            + W_MLM * loss_mlm
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if local_rank == 0:
            pbar.set_postfix({
                "Lg": f"{loss_global.item():.2f}",
                "Ll": f"{loss_local.item():.2f}",
                "Lmlm": f"{loss_mlm.item():.2f}",
            })

    if local_rank == 0:
        print(f"Epoch {epoch} complete")

dist.destroy_process_group()

