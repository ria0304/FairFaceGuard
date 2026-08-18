"""Week 3 -- standard training loop for the baseline detector.

Trains on your normal corpus (FF++ / Celeb-DF / DFDC etc.) with your normal
recipe. This is intentionally plain -- the research contribution is in
Weeks 1, 2, and 4, not in the baseline training procedure itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.detector.baseline_model import BaselineDeepfakeDetector


def train_baseline(
    model: BaselineDeepfakeDetector,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> BaselineDeepfakeDetector:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["fake_label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        msg = f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}"

        if val_loader is not None:
            val_acc = _evaluate_accuracy(model, val_loader, device)
            msg += f"  val_acc={val_acc:.4f}"

        print(msg)

    return model


@torch.no_grad()
def _evaluate_accuracy(model: BaselineDeepfakeDetector, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["fake_label"].to(device)
        preds = model(x).argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


if __name__ == "__main__":
    print(
        "This module expects a real DataLoader yielding "
        "{'image': Tensor[B,3,H,W], 'fake_label': Tensor[B]}. "
        "See src/data/datasets.py for the expected on-disk layout."
    )
