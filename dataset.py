import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class CXRPairedDataset(Dataset):
    def __init__(self, static_csv, dynamic_csv):
        # --------------------------------------------------
        # Load CSVs
        # --------------------------------------------------
        self.static_df = pd.read_csv(static_csv)
        self.dynamic_df = pd.read_csv(dynamic_csv)

        # --------------------------------------------------
        # SAFETY CHECK: row-by-row alignment
        # --------------------------------------------------
        assert len(self.static_df) == len(self.dynamic_df), \
            "Static and dynamic CSVs have different lengths"

        assert (
            self.static_df["prior_image_id"].values ==
            self.dynamic_df["prior_image_id"].values
        ).all(), "Mismatch in prior_image_id ordering"

        assert (
            self.static_df["current_image_id"].values ==
            self.dynamic_df["current_image_id"].values
        ).all(), "Mismatch in current_image_id ordering"

        # --------------------------------------------------
        # Combine into one dataframe
        # --------------------------------------------------
        self.df = self.static_df.copy()
        self.df["dynamic"] = self.dynamic_df["report"]

        # --------------------------------------------------
        # Image preprocessing (same as ImageEncoder)
        # --------------------------------------------------
        self.resize = T.Resize((512, 512))
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --------------------------------------------------
        # Load images
        # --------------------------------------------------
        curr = Image.open(row["current_path"]).convert("RGB")
        prev = Image.open(row["prior_path"]).convert("RGB")

        # --------------------------------------------------
        # Resize first (deterministic)
        # --------------------------------------------------
        curr = self.resize(curr)
        prev = self.resize(prev)

        # --------------------------------------------------
        # SHARED random crop (critical fix)
        # --------------------------------------------------
        i, j, h, w = T.RandomCrop.get_params(
            curr, output_size=(384, 384)
        )

        curr = TF.crop(curr, i, j, h, w)
        prev = TF.crop(prev, i, j, h, w)

        # --------------------------------------------------
        # To tensor + normalize
        # --------------------------------------------------
        curr = TF.to_tensor(curr)
        prev = TF.to_tensor(prev)

        curr = self.normalize(curr)
        prev = self.normalize(prev)

        # --------------------------------------------------
        # Text fields
        # --------------------------------------------------
        static_txt = row["report"]
        dynamic_txt = row["dynamic"]

        if not isinstance(static_txt, str) or static_txt.strip() == "":
            static_txt = "No findings."

        if not isinstance(dynamic_txt, str) or dynamic_txt.strip() == "":
            dynamic_txt = "No change."
        # --------------------------------------------------
        # Return sample
        # --------------------------------------------------
        return {
            "current_img": curr,
            "prior_img": prev,
            "static_text": static_txt,
            "dynamic_text": dynamic_txt,
        }

