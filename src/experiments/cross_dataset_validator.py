"""Train on one dataset (FF++), evaluate zero-shot on another (Celeb-DF).

This is the experiment that tells you whether the skin-tone/illumination
subgroup gap found on FF++ is a property of deepfake detectors generally,
or an artifact of FF++'s specific manipulation methods. AI-Face never
supported this: it has no natural second dataset with matching
labels.csv/annotations.csv structure and a disjoint identity pool. FF++ +
Celeb-DF do, via `ffplus_adapter.py` / `celebdf_adapter.py` writing the
same schema.

Celeb-DF is evaluation-only here (see `celebdf_adapter.py` docstring) --
this module never trains on it, only scores the FF++-trained model on it.

Usage:
    python -m src.experiments.cross_dataset_validator \\
        --train_data_root ./data/ffplus_processed \\
        --test_data_root ./data/celebdf_processed \\
        --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.data.datasets import AnnotatedFaceDataset
from src.detector.baseline_model import BaselineDeepfakeDetector
from src.detector.train_baseline import train_baseline
from src.detector.subgroup_eval import run_subgroup_eval
from src.utils.seed import set_seed


def cross_dataset_experiment(
    train_data_root: str,
    test_data_root: str,
    epochs: int = 20,
    backbone: str = "efficientnet_b4",
    batch_size: int = 32,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """Train baseline on train_data_root (FF++), evaluate in-domain on its
    own held-out test split AND out-of-domain (zero-shot) on
    test_data_root (Celeb-DF). Reports the generalization drop in overall
    AUC plus both datasets' subgroup gap tables side by side, so you can
    see whether the SAME subgroup (e.g. Fitzpatrick V/VI) is disadvantaged
    in both, or whether the gap pattern is dataset-specific."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_ds = AnnotatedFaceDataset(train_data_root, split="train")
    val_ds = AnnotatedFaceDataset(train_data_root, split="val")
    in_domain_test_ds = AnnotatedFaceDataset(train_data_root, split="test")

    # Celeb-DF is eval-only (every row is split="test", see celebdf_adapter.py)
    # so it has no train rows of its own to derive illumination-bin boundaries
    # from. AnnotatedFaceDataset refuses to compute boundaries from a
    # train-less data_root (RESEARCH_AUDIT.md item 3) -- copy the FF++
    # boundaries over so Celeb-DF's illum_bin uses the same, train-only-
    # derived boundaries rather than silently falling back to leaky
    # whole-dataset binning.
    import shutil

    from src.data.datasets import BOUNDARIES_FILENAME

    train_boundaries_path = os.path.join(train_data_root, BOUNDARIES_FILENAME)
    test_boundaries_path = os.path.join(test_data_root, BOUNDARIES_FILENAME)
    if os.path.exists(train_boundaries_path) and not os.path.exists(test_boundaries_path):
        shutil.copy(train_boundaries_path, test_boundaries_path)

    out_domain_ds = AnnotatedFaceDataset(test_data_root, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    in_domain_loader = DataLoader(in_domain_test_ds, batch_size=batch_size, num_workers=4)
    out_domain_loader = DataLoader(out_domain_ds, batch_size=batch_size, num_workers=4)

    model = BaselineDeepfakeDetector(backbone_name=backbone)
    model = train_baseline(model, train_loader, val_loader, epochs=epochs, device=device)

    in_domain = run_subgroup_eval(model, in_domain_loader, device=device)
    out_domain = run_subgroup_eval(model, out_domain_loader, device=device)

    in_domain_auc = _overall_auc(in_domain)
    out_domain_auc = _overall_auc(out_domain)

    report = {
        "train_dataset": train_data_root,
        "test_dataset": test_data_root,
        "in_domain_overall_auc": in_domain_auc,
        "out_domain_overall_auc": out_domain_auc,
        "generalization_drop": in_domain_auc - out_domain_auc,
        "in_domain_gap_table": {
            "by_skin_tone": in_domain["by_skin_tone"],
            "by_illumination": in_domain["by_illumination"],
        },
        "out_domain_gap_table": {
            "by_skin_tone": out_domain["by_skin_tone"],
            "by_illumination": out_domain["by_illumination"],
        },
    }
    return report


def _overall_auc(subgroup_eval_results: dict) -> float:
    """Sample-weighted average AUC across skin-tone bins, as a stand-in for
    an overall-dataset AUC (run_subgroup_eval doesn't compute an
    un-grouped AUC directly, only per-bin)."""
    by_skin = {k: v for k, v in subgroup_eval_results["by_skin_tone"].items() if k != "__gap__"}
    total_n = sum(v["n"] for v in by_skin.values())
    if total_n == 0:
        return float("nan")
    return sum(v["auc"] * v["n"] for v in by_skin.values()) / total_n


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_data_root", required=True, help="FF++ processed data_root (trained on)")
    p.add_argument("--test_data_root", required=True, help="Celeb-DF processed data_root (zero-shot eval only)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--backbone", default="efficientnet_b4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Path to write JSON report (default: <train_data_root>/cross_dataset_report.json)")
    return p


def main():
    args = build_arg_parser().parse_args()
    report = cross_dataset_experiment(
        train_data_root=args.train_data_root,
        test_data_root=args.test_data_root,
        epochs=args.epochs,
        backbone=args.backbone,
        seed=args.seed,
    )
    output_path = args.output or f"{args.train_data_root.rstrip('/')}/cross_dataset_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[cross_dataset_validator] in-domain AUC={report['in_domain_overall_auc']:.4f}  "
          f"out-of-domain AUC={report['out_domain_overall_auc']:.4f}  "
          f"drop={report['generalization_drop']:.4f}")
    print(f"[cross_dataset_validator] report written to {output_path}")


if __name__ == "__main__":
    # _overall_auc's weighting logic is pure and torch-free -- smoke test
    # it directly with a fabricated run_subgroup_eval-shaped dict. The
    # full cross_dataset_experiment() path needs torch + real data.
    fake_by_skin = {
        "I": {"n": 100, "auc": 0.90},
        "III": {"n": 50, "auc": 0.80},
        "__gap__": {"accuracy": 0.1},
    }
    result = {"by_skin_tone": fake_by_skin}
    auc = _overall_auc(result)
    expected = (0.90 * 100 + 0.80 * 50) / 150
    assert abs(auc - expected) < 1e-9, f"{auc} != {expected}"
    print(f"[smoke test] weighted AUC = {auc:.4f} (expected {expected:.4f})")
    print("[smoke test] PASSED")
