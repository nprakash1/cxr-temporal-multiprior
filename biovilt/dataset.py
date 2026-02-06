# dataset.py
import random
from pathlib import Path
from typing import Optional, List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ============================================================
# BASE IMAGE TRANSFORM (DETERMINISTIC)
# ============================================================
BASE_TRANSFORM = T.Compose([
    T.Resize(512),
    T.CenterCrop(448),
])


# ============================================================
# SYNCED AUGMENTATION SAMPLING
# ============================================================
def sample_augmentation(train: bool):
    """
    Sample augmentation parameters ONCE.
    """
    if not train:
        return None

    return {
        "angle": random.uniform(-30, 30),
        "shear": random.uniform(-15, 15),
        "brightness": random.uniform(0.8, 1.2),
        "contrast": random.uniform(0.8, 1.2),
    }


def apply_augmentation(img: Image.Image, params):
    """
    Apply identical augmentation params to an image.
    """
    if params is not None:
        img = TF.affine(
            img,
            angle=params["angle"],
            translate=(0, 0),
            scale=1.0,
            shear=[params["shear"], 0.0],
        )
        img = TF.adjust_brightness(img, params["brightness"])
        img = TF.adjust_contrast(img, params["contrast"])

    img = TF.to_tensor(img)

    # Ensure 3 channels
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    return img


# ============================================================
# IMAGE PATH RESOLUTION (MIMIC-CXR-JPG)
# ============================================================
def resolve_image_path(
    base_dir: Path,
    subject_id: int,
    study_id: int,
    dicom_id: str,
) -> Path:
    pid = str(subject_id)
    return (
        base_dir
        / f"p{pid[:2]}"
        / f"p{pid}"
        / f"s{study_id}"
        / f"{dicom_id}.jpg"
    )


# ============================================================
# DATASET
# ============================================================
class BioViLTDataset(Dataset):
    """
    Paper-faithful dataset:
      - Ds: current only, prior=None
      - Dm: current + prior
      - EXACT SAME transforms for current & prior
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        split: str,
        train: bool,
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.image_root = Path(image_root)
        self.train = train

        # Split indices
        self.single_indices = self.df.index[self.df["has_prior"] == False].tolist()
        self.multi_indices  = self.df.index[self.df["has_prior"] == True].tolist()

    def __len__(self):
        return len(self.df)

    def _load_raw(self, dicom_id, subject_id, study_id) -> Image.Image:
        path = resolve_image_path(self.image_root, subject_id, study_id, dicom_id)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        subject_id = int(row["subject_id"])
        study_id   = int(row["study_id"])

        # ---- load raw images ----
        curr_raw = self._load_raw(
            row["dicom_id_curr"],
            subject_id,
            study_id,
        )

        prior_raw = None
        if row["has_prior"]:
            prior_raw = self._load_raw(
                row["dicom_id_prior"],
                subject_id,
                int(row["prior_study_id"]),
            )

        # ---- deterministic base ----
        curr_raw = BASE_TRANSFORM(curr_raw)
        if prior_raw is not None:
            prior_raw = BASE_TRANSFORM(prior_raw)

        # ---- synced augmentation ----
        params = sample_augmentation(self.train)

        curr_img = apply_augmentation(curr_raw, params)
        prior_img = (
            apply_augmentation(prior_raw, params)
            if prior_raw is not None
            else None
        )

        return {
            "current_image": curr_img,
            "prior_image": prior_img,  # None for Ds
            "has_prior": bool(row["has_prior"]),
            "text": row["full_report_text"],
        }


# ============================================================
# HOMOGENEOUS BATCH SAMPLER (Ds / Dm)
# ============================================================
class BioViLTMixedBatchSampler(Sampler[List[int]]):
    """
    Each batch is entirely Ds or entirely Dm.
    All samples used exactly once per epoch.
    """

    def __init__(self, dataset: BioViLTDataset, batch_size: int, seed: int = 0):
        self.ds = dataset.single_indices
        self.dm = dataset.multi_indices
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)

        ds = self.ds.copy()
        dm = self.dm.copy()

        rng.shuffle(ds)
        rng.shuffle(dm)

        ds_batches = [ds[i:i+self.batch_size] for i in range(0, len(ds), self.batch_size)]
        dm_batches = [dm[i:i+self.batch_size] for i in range(0, len(dm), self.batch_size)]

        batches = ds_batches + dm_batches
        rng.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        return (len(self.ds) + len(self.dm)) // self.batch_size


# ============================================================
# COLLATE FUNCTION
# ============================================================
def biovilt_collate_fn(batch):
    has_prior = batch[0]["has_prior"]

    curr = torch.stack([b["current_image"] for b in batch])
    prior = (
        torch.stack([b["prior_image"] for b in batch])
        if has_prior
        else None
    )

    return {
        "current_image": curr,
        "prior_image": prior,
        "has_prior": has_prior,
        "text": [b["text"] for b in batch],
    }


# ============================================================
# SANITY CHECK
# ============================================================
if __name__ == "__main__":
    ds = BioViLTDataset(
        csv_path="biovilt_pretrain_train_imagelevel.csv",
        image_root="/scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0/files",
        split="train",
        train=True,
    )

    print("Total:", len(ds))
    print("Ds:", len(ds.single_indices))
    print("Dm:", len(ds.multi_indices))

    # Check transform sync
    idx = ds.multi_indices[0]
    sample = ds[idx]

    diff = (sample["current_image"] - sample["prior_image"]).abs().mean()
    print("Mean |current - prior| after identical transform:", diff.item())

