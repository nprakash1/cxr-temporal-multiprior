import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from dataset import CXRPairedDataset
from tempcxr.modules.tempcxr_model import TempCXR
from tempcxr.modules.image_encoder import ImageEncoder
from tempcxr.modules.text_encoder import TextEncoder
from tempcxr.modules.cross_exam_encoder import CrossExamEncoder
from losses import info_nce


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
# PATHS
# ============================================================
CKPT_DIR = "/scratch/m000081/eprakash/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

RESUME_CKPT = CKPT_DIR + "/tempa_epoch_29.pt"


# ============================================================
# HYPERPARAMETERS
# ============================================================
LR = 3e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
EPOCHS = 50
WARMUP = 1000
PROJ_DIM = 128

ALPHA = 1.0
BETA = 1.0


# ============================================================
# LOGGING
# ============================================================
if local_rank == 0:
    log_file = open("training_log_allgather.txt", "a")
else:
    log_file = None


# ============================================================
# BUILD MODEL
# ============================================================
model = TempCXR(
    text_encoder=TextEncoder(),
    image_encoder=ImageEncoder(),
    cross_encoder=CrossExamEncoder(),
    proj_dim=PROJ_DIM,
).to(DEVICE)

model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=True,
)


# ============================================================
# FREEZE TEXT ENCODER
# ============================================================
for p in model.module.text_encoder.parameters():
    p.requires_grad = False


# ============================================================
# TEMPERATURE PARAMETERS
# ============================================================
logit_scale_static  = nn.Parameter(torch.tensor(0.0, device=DEVICE))
logit_scale_dynamic = nn.Parameter(torch.tensor(0.0, device=DEVICE))


# ============================================================
# OPTIMIZER
# ============================================================
optimizer = AdamW(
    list(model.module.image_encoder.parameters()) +
    list(model.module.cross_encoder.parameters()) +
    list(model.module.proj_img_static.parameters()) +
    list(model.module.proj_img_dynamic.parameters()) +
    [logit_scale_static, logit_scale_dynamic],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# AMP
# ============================================================
scaler = torch.cuda.amp.GradScaler()


# ============================================================
# DATASET + SAMPLER
# ============================================================
dataset = CXRPairedDataset(
    static_csv="static_reports.csv",
    dynamic_csv="dynamic_reports.csv",
)

sampler = DistributedSampler(dataset)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=8,
    pin_memory=True,
)


# ============================================================
# SCHEDULER
# ============================================================
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP,
    num_training_steps=len(loader) * EPOCHS,
)


# ============================================================
# OPTIONAL RESUME
# ============================================================
start_epoch = 1
if RESUME_CKPT is not None:
    ckpt = torch.load(RESUME_CKPT, map_location={"cuda:0": f"cuda:{local_rank}"})
    model.module.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    logit_scale_static.data.copy_(ckpt["logit_scale_static"])
    logit_scale_dynamic.data.copy_(ckpt["logit_scale_dynamic"])
    start_epoch = ckpt["epoch"] + 1

    if local_rank == 0:
        print(f"✅ Resumed from epoch {ckpt['epoch']} → starting at {start_epoch}")


# ============================================================
# GRADIENT-PRESERVING ALL-GATHER
# ============================================================
class GatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        outputs = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(outputs, tensor)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        batch = grad_output.size(0) // ctx.world_size
        start = ctx.rank * batch
        end = start + batch
        return grad_output[start:end]


def gather_with_grad(tensor):
    return GatherWithGrad.apply(tensor)


# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(start_epoch, EPOCHS + 1):

    sampler.set_epoch(epoch)

    running_loss = 0.0
    running_static = 0.0
    running_dynamic = 0.0
    num_batches = 0

    if local_rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
    else:
        pbar = loader

    for batch in pbar:
        curr = batch["current_img"].to(DEVICE)
        prev = batch["prior_img"].to(DEVICE)
        static_text = batch["static_text"]
        dynamic_text = batch["dynamic_text"]

        with torch.cuda.amp.autocast():
            vs, ts, vd, td, _ = model(curr, prev, static_text, dynamic_text)

            vs_all = gather_with_grad(vs)
            ts_all = gather_with_grad(ts)
            vd_all = gather_with_grad(vd)
            td_all = gather_with_grad(td)

            # <<< ADDED: ONE-TIME ALL-GATHER SHAPE CHECK >>>
            if local_rank == 0 and epoch == start_epoch and num_batches == 0:
                print(
                    f"[ALL-GATHER CHECK] "
                    f"vs local={vs.shape}, "
                    f"vs_all={vs_all.shape}, "
                    f"expected={BATCH_SIZE * WORLD_SIZE}"
                )

            loss_static = info_nce(vs_all, ts_all, logit_scale_static.exp())
            loss_dynamic = info_nce(vd_all, td_all, logit_scale_dynamic.exp())
            loss = ALPHA * loss_dynamic + BETA * loss_static

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        running_static += loss_static.item()
        running_dynamic += loss_dynamic.item()
        num_batches += 1

        if local_rank == 0:
            pbar.set_postfix({
                "Ls": f"{loss_static.item():.4f}",
                "Ld": f"{loss_dynamic.item():.4f}",
                "L":  f"{loss.item():.4f}",
            })

    if local_rank == 0:
        epoch_loss = running_loss / num_batches
        epoch_static = running_static / num_batches
        epoch_dynamic = running_dynamic / num_batches

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Total={epoch_loss:.4f} | "
            f"Static={epoch_static:.4f} | "
            f"Dynamic={epoch_dynamic:.4f}"
        )

        log_file.write(
            f"{epoch},{epoch_loss},{epoch_static},{epoch_dynamic}\n"
        )
        log_file.flush()

        ckpt_path = os.path.join(CKPT_DIR, f"tempa_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "logit_scale_static": logit_scale_static.data,
            "logit_scale_dynamic": logit_scale_dynamic.data,
        }, ckpt_path)


if local_rank == 0:
    log_file.close()
    print("✅ Training complete.")

dist.destroy_process_group()

