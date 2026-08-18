"""Week 4 -- disentanglement training + lambda sweep.

Runs the adversarial-invariance experiment (methodology Section 5.1):
trains one model per (lambda_skin, lambda_illum) combination on independent
grids, so you can trace two separate accuracy-vs-invariance frontiers rather
than one confounded curve. lambda=0 for both recovers the Week-3 baseline
as a sanity check.
"""

from __future__ import annotations

import itertools

import torch
from torch.utils.data import DataLoader

from src.disentangle.grl import dann_lambda_schedule
from src.disentangle.losses import DisentangleLossWeights, disentangle_loss
from src.disentangle.model import DisentangledDeepfakeDetector


def train_one_config(
    lambda_skin_max: float,
    lambda_illum_max: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    backbone_name: str = "efficientnet_b4",
    baseline_state_dict: dict | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Trains a single model at the given (lambda_skin_max, lambda_illum_max)
    setpoint, warming lambda up via the DANN schedule within each run.
    Returns the trained model plus final val-set accuracy / probe accuracy,
    which together define one point on each frontier."""
    model = DisentangledDeepfakeDetector(backbone_name=backbone_name, pretrained=baseline_state_dict is None)
    if baseline_state_dict is not None:
        model.load_backbone_from_baseline(baseline_state_dict)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    weights = DisentangleLossWeights()
    total_steps = epochs * len(train_loader)
    step = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            progress = step / max(total_steps, 1)
            warm = dann_lambda_schedule(progress)
            model.set_lambdas(lambda_skin_max * warm, lambda_illum_max * warm)

            x = batch["image"].to(device)
            outputs = model(x)
            losses = disentangle_loss(
                outputs,
                batch["fake_label"].to(device),
                batch["skin_bin"].to(device),
                batch["illum_bin"].to(device),
                weights,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            step += 1

        print(f"  [lambda_skin={lambda_skin_max} lambda_illum={lambda_illum_max}] "
              f"epoch {epoch + 1}/{epochs} total_loss={losses['total'].item():.4f}")

    val_metrics = _evaluate(model, val_loader, device)
    return {"model": model, "lambda_skin": lambda_skin_max, "lambda_illum": lambda_illum_max, **val_metrics}


@torch.no_grad()
def _evaluate(model: DisentangledDeepfakeDetector, loader: DataLoader, device: str) -> dict:
    model.eval()
    correct_fake, correct_skin, correct_illum, total = 0, 0, 0, 0
    for batch in loader:
        x = batch["image"].to(device)
        out = model(x)
        total += x.size(0)
        correct_fake += (out["fake_logits"].argmax(-1).cpu() == batch["fake_label"]).sum().item()
        correct_skin += (out["skin_logits"].argmax(-1).cpu() == batch["skin_bin"]).sum().item()
        correct_illum += (out["illum_logits"].argmax(-1).cpu() == batch["illum_bin"]).sum().item()
    return {
        "fake_acc": correct_fake / max(total, 1),
        "skin_probe_acc": correct_skin / max(total, 1),   # low = successfully invariant
        "illum_probe_acc": correct_illum / max(total, 1),
    }


def run_independent_sweep(
    train_loader: DataLoader,
    val_loader: DataLoader,
    skin_lambda_grid: list[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    illum_lambda_grid: list[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    epochs_per_run: int = 15,
    baseline_state_dict: dict | None = None,
) -> dict:
    """Two INDEPENDENT sweeps (not a joint grid): vary lambda_skin with
    lambda_illum=0, and vary lambda_illum with lambda_skin=0. This isolates
    each factor's individual accuracy-vs-invariance frontier, which is what
    the methodology's Section 5.1 interpretation depends on. Extend to a
    joint grid only if you also want to study their interaction."""
    skin_frontier = []
    for lam in skin_lambda_grid:
        result = train_one_config(
            lambda_skin_max=lam,
            lambda_illum_max=0.0,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs_per_run,
            baseline_state_dict=baseline_state_dict,
        )
        skin_frontier.append({k: v for k, v in result.items() if k != "model"})

    illum_frontier = []
    for lam in illum_lambda_grid:
        result = train_one_config(
            lambda_skin_max=0.0,
            lambda_illum_max=lam,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs_per_run,
            baseline_state_dict=baseline_state_dict,
        )
        illum_frontier.append({k: v for k, v in result.items() if k != "model"})

    return {"skin_frontier": skin_frontier, "illum_frontier": illum_frontier}
