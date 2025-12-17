import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as T         # <-- ADD THIS

class CXRPairedDataset(torch.utils.data.Dataset):
    def __init__(self, static_csv, dynamic_csv):
        self.static_df = pd.read_csv(static_csv)
        self.dynamic_df = pd.read_csv(dynamic_csv)

        assert len(self.static_df) == len(self.dynamic_df)
        self.df = self.static_df.copy()
        self.df["dynamic"] = self.dynamic_df["report"]

        # ---- ADD EXACT TRANSFORM HERE (same as ImageEncoder) ----
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.RandomCrop((384, 384)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        curr = Image.open(row["current_path"]).convert("RGB")
        prev = Image.open(row["prior_path"]).convert("RGB")

        # ---- APPLY TRANSFORMS HERE (critical fix) ----
        curr = self.transform(curr)
        prev = self.transform(prev)

        static_txt = row["report"]
        dynamic_txt = row["dynamic"]

        if not isinstance(static_txt, str) or static_txt.strip() == "":
            static_txt = "no findings"

        if not isinstance(dynamic_txt, str) or dynamic_txt.strip() == "":
            dynamic_txt = "no change"

        return {
            "current_img": curr,       # now a Tensor
            "prior_img": prev,         # now a Tensor
            "static_text": static_txt,
            "dynamic_text": dynamic_txt
        }

