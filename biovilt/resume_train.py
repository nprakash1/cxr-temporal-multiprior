# resume_train.py

import os
import glob
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import BioViLTDataset, biovilt_collate_fn
from tempcxr.modules.tempcxr_model import TempCXR
from losses import global_contrastive_loss, local_contrastive_loss, mlm_loss
from migrate_checkpoint import migrate_state_dict


# ============================================================
# PATHS — overridable via env vars (or CLI). Defaults are the
# original Marlowe-cluster paths so cluster training is unchanged;
# Colab / local runs should set env vars or pass CLI flags.
# ============================================================
DEFAULT_CHECKPOINT_DIR = "/scratch/m000081/eprakash/temporal/checkpoints"
DEFAULT_CSV_DIR        = "/scratch/m000081/eprakash/temporal/model/biovilt"
DEFAULT_IMAGE_ROOT     = "/scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0/files"
DEFAULT_LOG_DIR        = "/scratch/m000081/eprakash/temporal/logs"


# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None,
                    help="Resume from a saved training checkpoint (epoch_N.pt). "
                         "If None, the latest in CHECKPOINT_DIR is used.")
parser.add_argument("--init-from", type=str, default=None,
                    help="Initialize from a raw weights file (e.g. upstream "
                         "BioViL-T) when no --resume checkpoint exists. "
                         "Auto-migrated to current --k-max.")
parser.add_argument("--k-max", type=int, default=1,
                    help="Maximum number of priors per sample. K_max=1 "
                         "reproduces original single-prior BioViL-T training. "
                         "Set 2..N to enable multi-prior joint self-attention.")
parser.add_argument("--mode", type=str, default="biovilt",
                    choices=["biovil", "biovilt", "biovilt_finetuned"],
                    help="Image-encoder weight init source for a fresh run.")
# Path overrides. Resolution order per path: CLI flag → env var → cluster default.
parser.add_argument("--image-root",     type=str, default=None,
                    help="Root dir of MIMIC-CXR-JPG files/ tree.")
parser.add_argument("--csv-dir",        type=str, default=None,
                    help="Dir containing biovilt_pretrain_*.csv (used only if "
                         "--train-csv / --val-csv aren't set explicitly).")
parser.add_argument("--train-csv",      type=str, default=None,
                    help="Explicit path to the training CSV. Overrides --csv-dir.")
parser.add_argument("--val-csv",        type=str, default=None,
                    help="Explicit path to the validation CSV. Overrides --csv-dir.")
parser.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Where to save epoch_*.pt and best.pt.")
parser.add_argument("--log-dir",        type=str, default=None,
                    help="Where to write val_metrics.csv.")
parser.add_argument("--epochs", type=int, default=None,
                    help="Override the default 50 epochs (useful for smoke runs).")
parser.add_argument("--batch-size", type=int, default=None,
                    help="Override the default per-GPU batch size of 32.")
args = parser.parse_args()


def _pick(cli_val, env_key, default):
    """Resolution: explicit CLI flag → env var → cluster default."""
    if cli_val is not None:
        return cli_val
    return os.environ.get(env_key, default)


CHECKPOINT_DIR = _pick(args.checkpoint_dir, "CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
CSV_DIR        = _pick(args.csv_dir,        "CSV_DIR",        DEFAULT_CSV_DIR)
IMAGE_ROOT     = _pick(args.image_root,     "IMAGE_ROOT",     DEFAULT_IMAGE_ROOT)
LOG_DIR        = _pick(args.log_dir,        "LOG_DIR",        DEFAULT_LOG_DIR)

TRAIN_CSV = args.train_csv or os.path.join(CSV_DIR, "biovilt_pretrain_train_imagelevel.csv")
VAL_CSV   = args.val_csv   or os.path.join(CSV_DIR, "biovilt_pretrain_combined_imagelevel.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_LOG = os.path.join(LOG_DIR, "val_metrics.csv")


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

# A full, verbose run-config banner is printed later (after the hyper-
# parameters and dataloaders are resolved) — see `print_run_config()`.



def ddp_reduce(value):
    tensor = torch.tensor(value, device=DEVICE)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return tensor.item()


# ============================================================
# GRADIENT-PRESERVING ALL-GATHER
# ============================================================
class GatherWithGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        tensor = tensor.contiguous()
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
# PAPER-FAITHFUL MIXED DISTRIBUTED BATCH SAMPLER
# ============================================================
class DistributedMixedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, batch_size, shuffle=True):
        self.single = dataset.single_indices
        self.multi = dataset.multi_indices
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.global_batch = batch_size * self.world_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        total = len(self.single) + len(self.multi)
        return total // self.global_batch

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.epoch)

        single = torch.tensor(self.single)
        multi = torch.tensor(self.multi)

        if self.shuffle:
            single = single[torch.randperm(len(single), generator=g)]
            multi = multi[torch.randperm(len(multi), generator=g)]

        single = single.tolist()
        multi = multi.tolist()

        sp = 0
        mp = 0
        total = len(single) + len(multi)
        p_single = len(single) / total

        while True:

            if sp + self.global_batch > len(single) and \
               mp + self.global_batch > len(multi):
                break

            if sp + self.global_batch > len(single):
                choose_single = False
            elif mp + self.global_batch > len(multi):
                choose_single = True
            else:
                choose_single = torch.rand(1, generator=g).item() < p_single

            if choose_single:
                batch = single[sp: sp + self.global_batch]
                sp += self.global_batch
            else:
                batch = multi[mp: mp + self.global_batch]
                mp += self.global_batch

            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield batch[start:end]


# ============================================================
# HYPERPARAMETERS
# ============================================================
LR = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE   = args.batch_size if args.batch_size is not None else 32
EPOCHS       = args.epochs     if args.epochs     is not None else 50
WARMUP_RATIO = 0.03

# DataLoader worker count: cluster default 8; Colab/local should set
# NUM_WORKERS=2 (free Colab only has 2 vCPU). Env var is the override.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))

W_GLOBAL = 1.0
W_LOCAL = 0.5
W_MLM = 1.0


# ============================================================
# DATASETS
# ============================================================
train_dataset = BioViLTDataset(
    csv_path=TRAIN_CSV,
    image_root=IMAGE_ROOT,
    split="train",
    train=True,
    k_max=args.k_max,
)

val_dataset = BioViLTDataset(
    csv_path=VAL_CSV,
    image_root=IMAGE_ROOT,
    split="val",
    train=False,
    k_max=args.k_max,
)

train_sampler = DistributedMixedBatchSampler(train_dataset, BATCH_SIZE, shuffle=True)
val_sampler = DistributedMixedBatchSampler(val_dataset, BATCH_SIZE, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=biovilt_collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=biovilt_collate_fn,
)


# ============================================================
# VERBOSE RUN-CONFIG BANNER
# ============================================================
def print_run_config():
    """
    Print, at the top of every run's log:
      1. a per-rank line proving each GPU is actually in use and showing
         the per-GPU batch size, and
      2. (rank 0 only) a full dump of every parameter, path, dataset size,
         and the per-GPU / global batch sizes.
    """
    # (1) Every rank announces the GPU it occupies + its per-GPU batch size.
    try:
        gpu_name = torch.cuda.get_device_name(local_rank)
    except Exception:
        gpu_name = "unknown"
    print(
        f"[gpu] rank {local_rank}/{WORLD_SIZE - 1} -> cuda:{local_rank} "
        f"({gpu_name}) | per-GPU batch_size={BATCH_SIZE}",
        flush=True,
    )

    # Make sure all per-rank lines are flushed before rank 0 prints the banner.
    dist.barrier()

    if local_rank != 0:
        return

    global_batch = BATCH_SIZE * WORLD_SIZE
    bar = "=" * 64
    print("\n" + bar)
    print(" RUN CONFIGURATION")
    print(bar)
    # --- model / training knobs ---
    print(f"[run]    mode            = {args.mode}")
    print(f"[run]    k_max           = {args.k_max}")
    print(f"[run]    epochs          = {EPOCHS}")
    print(f"[run]    resume          = {args.resume}")
    print(f"[run]    init_from       = {args.init_from}")
    # --- hyperparameters ---
    print(f"[hparam] lr              = {LR}")
    print(f"[hparam] weight_decay    = {WEIGHT_DECAY}")
    print(f"[hparam] warmup_ratio    = {WARMUP_RATIO}")
    print(f"[hparam] num_workers     = {NUM_WORKERS}")
    print(f"[hparam] loss weights    = global:{W_GLOBAL} local:{W_LOCAL} mlm:{W_MLM}")
    # --- paths ---
    print(f"[paths]  IMAGE_ROOT      = {IMAGE_ROOT}")
    print(f"[paths]  TRAIN_CSV       = {TRAIN_CSV}")
    print(f"[paths]  VAL_CSV         = {VAL_CSV}")
    print(f"[paths]  CHECKPOINT_DIR  = {CHECKPOINT_DIR}")
    print(f"[paths]  LOG_DIR         = {LOG_DIR}")
    # --- data ---
    print(f"[data]   train samples   = {len(train_dataset)}")
    print(f"[data]   val samples     = {len(val_dataset)}")
    print(f"[data]   steps/epoch     = {len(train_loader)}")
    # --- GPU / batch summary ---
    print("-" * 64)
    print(f"[gpus]   active GPUs (world_size) = {WORLD_SIZE}")
    print(f"[gpus]   per-GPU batch size       = {BATCH_SIZE}")
    print(f"[gpus]   GLOBAL batch size        = {BATCH_SIZE} x {WORLD_SIZE} = {global_batch}")
    print(bar + "\n", flush=True)


print_run_config()


# ============================================================
# MODEL
# ============================================================

model = TempCXR(mode=args.mode, K_max=args.k_max).to(DEVICE)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

num_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * num_steps),
    num_training_steps=num_steps,
)

scaler = torch.amp.GradScaler("cuda")

start_epoch = 1
best_val_loss = float("inf")


# ============================================================
# RESUME
# ============================================================
if args.resume is None:
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
    if checkpoints:
        args.resume = checkpoints[-1]

if args.resume is not None:
    checkpoint = torch.load(args.resume, map_location=DEVICE)
    # Migrate type_embed_multi if the checkpoint was saved at a different K_max.
    migrated_state, mig_log = migrate_state_dict(
        checkpoint["model"], K_max_new=args.k_max, verbose=False
    )
    model.module.load_state_dict(migrated_state)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    if local_rank == 0:
        print(f"Resumed from {args.resume} (K_max={args.k_max})")
        for line in mig_log:
            print(line)
elif args.init_from is not None:
    # Fresh run from raw weights (e.g. upstream BioViL-T) — auto-migrate.
    raw = torch.load(args.init_from, map_location=DEVICE)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    migrated_state, mig_log = migrate_state_dict(
        raw, K_max_new=args.k_max, verbose=False
    )
    missing, unexpected = model.module.load_state_dict(migrated_state, strict=False)
    if local_rank == 0:
        print(f"Initialized from {args.init_from} (K_max={args.k_max})")
        for line in mig_log:
            print(line)
        if missing:
            print(f"  [warn] {len(missing)} missing keys (kept random init)")
        if unexpected:
            print(f"  [warn] {len(unexpected)} unexpected keys (ignored)")


# ============================================================
# CSV HEADER
# ============================================================
if local_rank == 0 and not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w") as f:
        f.write("epoch,val_total,val_global,val_local,val_mlm\n")


# ============================================================
# TRAIN LOOP
# ============================================================
for epoch in range(start_epoch, EPOCHS + 1):

    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch)

    model.train()
    running_total = 0
    running_batches = 0

    if local_rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=120)
    else:
        pbar = train_loader

    for batch in pbar:

        curr = batch["current_image"].to(DEVICE)
        prior_imgs = batch["prior_images"]
        prior_mask = batch["prior_mask"]
        texts = batch["text"]

        if prior_imgs is not None:
            prior_imgs = prior_imgs.to(DEVICE)
            prior_mask = prior_mask.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(curr, prior_imgs, prior_mask, texts=texts)

            img_global_all = gather_with_grad(outputs["img_global"])
            txt_global_all = gather_with_grad(outputs["txt_global"])
            img_patches_all = gather_with_grad(outputs["img_patches"])
            txt_local_all = gather_with_grad(outputs["txt_local"])
            token_mask_all = gather_with_grad(outputs["token_mask"].float()).bool()

            loss_g = global_contrastive_loss(img_global_all, txt_global_all)
            loss_l = local_contrastive_loss(img_patches_all, txt_local_all, token_mask_all)
            loss_m = mlm_loss(outputs["mlm_logits"], outputs["mlm_labels"])

            loss = W_GLOBAL*loss_g + W_LOCAL*loss_l + W_MLM*loss_m

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_total += loss.item()
        running_batches += 1

        if local_rank == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{running_total/running_batches:.4f}"
            })

    if local_rank == 0:
        print(f"Train Epoch {epoch} | Avg Loss: {running_total/running_batches:.4f}")


    # ============================================================
    # VALIDATION
    # ============================================================
    model.eval()
    val_total = val_g = val_l = val_m = 0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:

            curr = batch["current_image"].to(DEVICE)
            prior_imgs = batch["prior_images"]
            prior_mask = batch["prior_mask"]
            texts = batch["text"]

            if prior_imgs is not None:
                prior_imgs = prior_imgs.to(DEVICE)
                prior_mask = prior_mask.to(DEVICE)

            with torch.amp.autocast("cuda"):
                outputs = model(curr, prior_imgs, prior_mask, texts=texts)

                loss_g = global_contrastive_loss(outputs["img_global"], outputs["txt_global"])
                loss_l = local_contrastive_loss(outputs["img_patches"], outputs["txt_local"], outputs["token_mask"])
                loss_m = mlm_loss(outputs["mlm_logits"], outputs["mlm_labels"])

                total = W_GLOBAL*loss_g + W_LOCAL*loss_l + W_MLM*loss_m

            val_total += total.item()
            val_g += loss_g.item()
            val_l += loss_l.item()
            val_m += loss_m.item()
            val_batches += 1

    val_total /= val_batches
    val_g /= val_batches
    val_l /= val_batches
    val_m /= val_batches

    val_total = ddp_reduce(val_total)
    val_g = ddp_reduce(val_g)
    val_l = ddp_reduce(val_l)
    val_m = ddp_reduce(val_m)

    if local_rank == 0:

        print(
            f"Val Epoch {epoch} | "
            f"Total={val_total:.4f} | "
            f"Global={val_g:.4f} | "
            f"Local={val_l:.4f} | "
            f"MLM={val_m:.4f}"
        )

        with open(CSV_LOG, "a") as f:
            f.write(f"{epoch},{val_total},{val_g},{val_l},{val_m}\n")

        ckpt = {
            "epoch": epoch,
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }

        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt"))

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print("🔥 Saved new BEST checkpoint")

dist.destroy_process_group()

