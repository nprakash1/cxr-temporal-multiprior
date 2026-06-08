# dataset.py

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
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
    Sample augmentation parameters ONCE per sample.
    The same params are applied to the current image and every prior image
    in that sample so spatial alignment between timesteps is preserved.
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
    Multi-prior dataset.

    Each sample:
      - current_image : Tensor (3, H, W)
      - prior_images  : list[Tensor (3, H, W)]   length 0..K_max  (newest first)
      - num_priors    : int in [0, K_max]
      - text          : str

    The same augmentation params are applied to the current image and every
    prior image, so spatial alignment across timesteps is preserved.

    Backward-compat aliases kept in the returned dict:
      - prior_image   : Tensor (3, H, W) or None    ← the newest prior, or None
      - has_prior     : bool

    Backward-compat CSV format:
      If the CSV has the legacy columns `dicom_id_prior` / `prior_study_id` /
      `has_prior` but not `num_priors`, the dataset transparently treats it
      as K_max=1 data (every sample has 0 or 1 priors).
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        split: str,
        train: bool,
        k_max: int = 4,
    ):
        assert k_max >= 1, "k_max must be >= 1"
        self.k_max = k_max

        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.image_root = Path(image_root)
        self.train = train

        # ---- Auto-detect CSV format ----
        # Prefer the new multi-prior columns when present; otherwise fall
        # back to the legacy single-prior columns.
        self._has_multi_cols = (
            "num_priors" in self.df.columns
            and "dicom_ids_prior" in self.df.columns
            and "prior_study_ids" in self.df.columns
        )
        self._has_legacy_cols = (
            "has_prior" in self.df.columns
            and "dicom_id_prior" in self.df.columns
            and "prior_study_id" in self.df.columns
        )
        if not (self._has_multi_cols or self._has_legacy_cols):
            raise ValueError(
                "CSV must contain either the new multi-prior columns "
                "(num_priors, dicom_ids_prior, prior_study_ids) or the "
                "legacy single-prior columns (has_prior, dicom_id_prior, "
                "prior_study_id)."
            )

        # ---- Precompute split indices for Ds (no prior) and Dm (>=1 prior) ----
        if self._has_multi_cols:
            num_priors_col = self.df["num_priors"].astype(int)
        else:
            num_priors_col = self.df["has_prior"].astype(int)
        self.single_indices = self.df.index[num_priors_col == 0].tolist()
        self.multi_indices  = self.df.index[num_priors_col > 0].tolist()

    # ---------- helpers ----------
    def __len__(self):
        return len(self.df)

    def _load_raw(self, dicom_id, subject_id, study_id) -> Image.Image:
        path = resolve_image_path(self.image_root, subject_id, study_id, dicom_id)
        return Image.open(path).convert("RGB")

    def _row_priors(self, row) -> List[Tuple[str, int]]:
        """
        Return a list of (dicom_id, study_id) prior tuples (newest first),
        truncated to self.k_max.
        """
        if self._has_multi_cols:
            n = int(row["num_priors"])
            if n == 0:
                return []
            dicom_ids = json.loads(row["dicom_ids_prior"])
            study_ids = json.loads(row["prior_study_ids"])
            assert len(dicom_ids) == len(study_ids) == n, (
                f"num_priors={n} but dicom_ids_prior/prior_study_ids have "
                f"len {len(dicom_ids)}/{len(study_ids)}"
            )
            return [(str(d), int(s)) for d, s in zip(dicom_ids, study_ids)][: self.k_max]
        else:
            # Legacy single-prior CSV
            if not bool(row["has_prior"]):
                return []
            return [(str(row["dicom_id_prior"]), int(row["prior_study_id"]))]

    # ---------- main ----------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        subject_id = int(row["subject_id"])
        study_id   = int(row["study_id"])

        # ---- Load current image ----
        curr_raw = self._load_raw(
            row["dicom_id_curr"],
            subject_id,
            study_id,
        )

        # ---- Load up to K_max prior images (newest first) ----
        prior_specs = self._row_priors(row)
        prior_raws = [
            self._load_raw(d_id, subject_id, s_id)
            for (d_id, s_id) in prior_specs
        ]

        # ---- Deterministic base transform (PIL) ----
        curr_raw = BASE_TRANSFORM(curr_raw)
        prior_raws = [BASE_TRANSFORM(p) for p in prior_raws]

        # ---- Synced augmentation (same params for curr + all priors) ----
        params = sample_augmentation(self.train)
        curr_img = apply_augmentation(curr_raw, params)
        prior_imgs = [apply_augmentation(p, params) for p in prior_raws]

        # ---- Backward-compat aliases ----
        prior_image_legacy = prior_imgs[0] if len(prior_imgs) > 0 else None
        has_prior_legacy   = len(prior_imgs) > 0

        return {
            # New multi-prior interface
            "current_image": curr_img,            # (3, H, W)
            "prior_images":  prior_imgs,          # list of (3, H, W), len 0..K_max
            "num_priors":    len(prior_imgs),     # int
            "text":          row["full_report_text"],

            # Backward-compat aliases (= newest prior, if any)
            "prior_image":   prior_image_legacy,
            "has_prior":     has_prior_legacy,
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================
def biovilt_collate_fn(batch):
    """
    Variable-K collate.

    Pads priors to ``K_batch = max(num_priors over batch)`` and builds a
    ``(B, K_batch)`` bool ``prior_mask`` (True = real prior slot, False = pad).

    Returned dict:

        current_image : Tensor (B, 3, H, W)
        prior_images  : Tensor (B, K_batch, 3, H, W) or None  (None iff every sample has 0 priors)
        prior_mask    : Tensor (B, K_batch) bool       or None
        num_priors    : LongTensor (B,)

        # Backward-compat aliases for the single-prior model:
        prior_image   : Tensor (B, 3, H, W) or None    (= prior_images[:, 0] when every sample has ≥1 prior; else None)
        has_prior     : bool                            (True iff every sample has ≥1 prior)

        text          : list[str]

    The legacy single-prior trainer keeps working as long as it builds
    homogeneous batches (which the existing
    ``Subset(ds, ds.single_indices) / Subset(ds, ds.multi_indices)`` pattern
    already does).
    """
    B = len(batch)
    curr = torch.stack([b["current_image"] for b in batch])  # (B, 3, H, W)

    num_priors_per = [int(b["num_priors"]) for b in batch]
    K_batch = max(num_priors_per) if num_priors_per else 0

    if K_batch == 0:
        prior_images       = None
        prior_mask         = None
        prior_image_legacy = None
        has_prior_legacy   = False
    else:
        _, C, H, W = curr.shape
        prior_images = torch.zeros(B, K_batch, C, H, W, dtype=curr.dtype)
        prior_mask   = torch.zeros(B, K_batch, dtype=torch.bool)
        for i, b in enumerate(batch):
            k_i = num_priors_per[i]
            for j in range(k_i):
                prior_images[i, j] = b["prior_images"][j]
            prior_mask[i, :k_i] = True

        # Legacy alias is only valid when EVERY sample contributes a prior_0,
        # i.e. the batch is homogeneous in "has prior?" — matches the legacy
        # contract that the single-prior trainer relied on.
        if all(n > 0 for n in num_priors_per):
            prior_image_legacy = prior_images[:, 0].clone()
            has_prior_legacy   = True
        else:
            prior_image_legacy = None
            has_prior_legacy   = False

    return {
        # ---- New multi-prior interface ----
        "current_image": curr,
        "prior_images":  prior_images,                                 # (B, K_batch, 3, H, W) or None
        "prior_mask":    prior_mask,                                   # (B, K_batch) bool   or None
        "num_priors":    torch.tensor(num_priors_per, dtype=torch.long),

        # ---- Backward-compat aliases ----
        "prior_image":   prior_image_legacy,                           # (B, 3, H, W) or None
        "has_prior":     has_prior_legacy,                             # bool

        # ---- Text ----
        "text":          [b["text"] for b in batch],
    }


# ============================================================
# SANITY CHECK
# ============================================================
if __name__ == "__main__":

    from torch.utils.data import Subset, DataLoader

    ds = BioViLTDataset(
        csv_path="biovilt_pretrain_train_imagelevel.csv",
        image_root="/scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0/files",
        split="train",
        train=True,
        k_max=4,
    )

    print("Total samples:", len(ds))
    print("Single (Ds):", len(ds.single_indices))
    print("Multi  (Dm):", len(ds.multi_indices))
    print("CSV format detected:",
          "multi-prior" if ds._has_multi_cols else "legacy single-prior")

    # ---- Check transform synchronization ----
    if len(ds.multi_indices) > 0:
        idx = ds.multi_indices[0]
        sample = ds[idx]
        print(f"Sample[{idx}]: num_priors = {sample['num_priors']}")
        print(f"  current_image: {tuple(sample['current_image'].shape)}")
        for j, p in enumerate(sample["prior_images"]):
            print(f"  prior_images[{j}]: {tuple(p.shape)}")
        # Mean diff between curr and newest prior (after identical transform)
        if sample["prior_image"] is not None:
            diff = (sample["current_image"] - sample["prior_image"]).abs().mean()
            print(f"  Mean |curr - newest_prior|: {diff.item():.4f}")

    # ---- Check subset creation (for distributed training) ----
    single_subset = Subset(ds, ds.single_indices)
    multi_subset  = Subset(ds, ds.multi_indices)
    print("Single subset size:", len(single_subset))
    print("Multi subset size :", len(multi_subset))

    # ---- Collate sanity: homogeneous and heterogeneous batches ----
    if len(ds.multi_indices) >= 4:
        # Homogeneous Dm batch
        batch_hom = [ds[i] for i in ds.multi_indices[:4]]
        out_hom = biovilt_collate_fn(batch_hom)
        print("\n[collate] Homogeneous Dm batch:")
        print(f"  current_image: {tuple(out_hom['current_image'].shape)}")
        print(f"  prior_images : "
              f"{tuple(out_hom['prior_images'].shape) if out_hom['prior_images'] is not None else None}")
        print(f"  prior_mask   : "
              f"{tuple(out_hom['prior_mask'].shape) if out_hom['prior_mask'] is not None else None}")
        print(f"  num_priors   : {out_hom['num_priors'].tolist()}")
        print(f"  has_prior    : {out_hom['has_prior']}  "
              f"(legacy alias non-None? {out_hom['prior_image'] is not None})")

    if len(ds.single_indices) >= 2 and len(ds.multi_indices) >= 2:
        # Heterogeneous batch: mix Ds + Dm
        batch_het = (
            [ds[i] for i in ds.single_indices[:2]]
            + [ds[i] for i in ds.multi_indices[:2]]
        )
        out_het = biovilt_collate_fn(batch_het)
        print("\n[collate] Heterogeneous batch:")
        print(f"  current_image: {tuple(out_het['current_image'].shape)}")
        print(f"  prior_images : "
              f"{tuple(out_het['prior_images'].shape) if out_het['prior_images'] is not None else None}")
        print(f"  prior_mask   : "
              f"{tuple(out_het['prior_mask'].shape) if out_het['prior_mask'] is not None else None}")
        print(f"  num_priors   : {out_het['num_priors'].tolist()}")
        print(f"  has_prior    : {out_het['has_prior']}  "
              f"(legacy alias non-None? {out_het['prior_image'] is not None})")

    print("\n✅ Dataset sanity check passed.")
