"""Expected on-disk layout and Dataset classes connecting Weeks 1-4.

    data_root/
      frames/<face_id>.png                        raw face crops
      annotations.csv                              from src/annotation (Week 1)
          columns: face_id, ita_continuous, fitzpatrick_bin, illuminant_*, flagged
      labels.csv                                   your ground-truth labels
          columns: face_id, fake_label (0/1)
      counterfactuals/<face_id>/{original,skin_only,illum_only,both}.png
                                                     from src/augmentation (Week 2)

Fitzpatrick bins are mapped to integers 0-5 (I-VI) and illumination is
quantile-binned into 5 bins from the illuminant estimate at load time.
Swap this module out entirely if your storage layout differs -- it is the
only place other modules assume a specific directory structure.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

FITZPATRICK_TO_INT = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5}

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AnnotatedFaceDataset(Dataset):
    """Used by Week 3 training/eval and Week 4 probing. Merges labels.csv
    with annotations.csv on face_id, and quantile-bins illuminant magnitude
    into 5 illumination bins."""

    def __init__(self, data_root: str, transform=DEFAULT_TRANSFORM, split: str | None = None):
        self.data_root = data_root
        self.transform = transform

        labels = pd.read_csv(os.path.join(data_root, "labels.csv"))
        annotations = pd.read_csv(os.path.join(data_root, "annotations.csv"))
        df = labels.merge(annotations, on="face_id", how="inner")
        df = df[~df.get("flagged", False)]

        df["skin_bin"] = df["fitzpatrick_bin"].map(FITZPATRICK_TO_INT)
        illum_mag = df[["illuminant_estimate_r", "illuminant_estimate_g", "illuminant_estimate_b"]].mean(axis=1) \
            if "illuminant_estimate_r" in df.columns else pd.Series(np.random.rand(len(df)))
        df["illum_bin"] = pd.qcut(illum_mag, 5, labels=False, duplicates="drop")

        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, "frames", f"{row['face_id']}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "fake_label": torch.tensor(int(row["fake_label"]), dtype=torch.long),
            "skin_bin": torch.tensor(int(row["skin_bin"]), dtype=torch.long),
            "illum_bin": torch.tensor(int(row["illum_bin"]), dtype=torch.long),
            "face_id": row["face_id"],
        }


class CounterfactualDataset(Dataset):
    """Used by Week 4's counterfactual_eval.py. Yields the factorial
    {original, skin_only, illum_only, both} image set per face."""

    def __init__(self, data_root: str, face_ids: list[str], transform=DEFAULT_TRANSFORM):
        self.data_root = data_root
        self.face_ids = face_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.face_ids)

    def _load(self, face_id: str, variant: str) -> torch.Tensor:
        path = os.path.join(self.data_root, "counterfactuals", face_id, f"{variant}.png")
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform else transforms.ToTensor()(image)

    def __getitem__(self, idx: int) -> dict:
        face_id = self.face_ids[idx]
        return {
            "face_id": face_id,
            "original": self._load(face_id, "original"),
            "skin_only": self._load(face_id, "skin_only"),
            "illum_only": self._load(face_id, "illum_only"),
            "both": self._load(face_id, "both"),
        }


def collate_counterfactual_batch(batch: list[dict]):
    """Turns a list of per-face dicts into the CounterfactualBatch shape
    expected by src/disentangle/counterfactual_eval.py."""
    from src.disentangle.counterfactual_eval import CounterfactualBatch

    return CounterfactualBatch(
        original=torch.stack([b["original"] for b in batch]),
        skin_only=torch.stack([b["skin_only"] for b in batch]),
        illum_only=torch.stack([b["illum_only"] for b in batch]),
        both=torch.stack([b["both"] for b in batch]),
        face_ids=[b["face_id"] for b in batch],
    )
