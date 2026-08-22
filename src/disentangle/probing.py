"""Week 4 -- diagnostic linear probing (methodology Section 5.2).

Run on the FROZEN Week-3 baseline (no adversarial training). High probe
accuracy means the representation makes that factor available/decodable --
necessary but not sufficient evidence the detector's actual real/fake
decision USES it. Pair with counterfactual_eval.py (Section 5.3) which
tests causal use directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


@torch.no_grad()
def extract_embeddings(frozen_model, loader: DataLoader, device: str) -> tuple[torch.Tensor, dict]:
    """frozen_model must expose .encode(x) -> z (both BaselineDeepfakeDetector
    and DisentangledDeepfakeDetector provide this)."""
    frozen_model.eval().to(device)
    zs, fake_labels, skin_labels, illum_labels = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        z = frozen_model.encode(x)
        zs.append(z.cpu())
        fake_labels.append(batch["fake_label"])
        skin_labels.append(batch["skin_bin"])
        illum_labels.append(batch["illum_bin"])
    return torch.cat(zs), {
        "fake_label": torch.cat(fake_labels),
        "skin_bin": torch.cat(skin_labels),
        "illum_bin": torch.cat(illum_labels),
    }


def train_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    epochs: int = 50,
    lr: float = 1e-2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Trains a single linear probe on frozen embeddings and returns its
    validation accuracy -- the "leakage" number reported in the paper."""
    probe = LinearProbe(z_train.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    z_train, y_train = z_train.to(device), y_train.to(device)
    z_val, y_val = z_val.to(device), y_val.to(device)

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(z_train), y_train)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_acc = (probe(z_val).argmax(-1) == y_val).float().mean().item()

    return {"val_accuracy": val_acc, "probe": probe}


def run_leakage_report(frozen_model, train_loader, val_loader, n_skin_bins=6, n_illum_bins=5, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    z_tr, labels_tr = extract_embeddings(frozen_model, train_loader, device)
    z_va, labels_va = extract_embeddings(frozen_model, val_loader, device)

    skin_result = train_probe(z_tr, labels_tr["skin_bin"], z_va, labels_va["skin_bin"], n_skin_bins, device=device)
    illum_result = train_probe(z_tr, labels_tr["illum_bin"], z_va, labels_va["illum_bin"], n_illum_bins, device=device)

    return {
        "skin_probe_accuracy": skin_result["val_accuracy"],
        "illum_probe_accuracy": illum_result["val_accuracy"],
        "note": "Higher = more leakage of that factor into the frozen baseline's representation.",
    }
