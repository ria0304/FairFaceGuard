"""Sweep the baseline detector across backbones and check whether the
Delta_skin > Delta_illum finding (or its reverse) holds regardless of
architecture, or is an artifact of one backbone's inductive biases.

`src/detector/baseline_model.build_backbone()` already supports
EfficientNet-B4 (the locked primary), Xception, and any timm ViT/CLIP
name -- this module just loops over that registry and reuses the exact
Week 3 (`train_baseline`, `run_subgroup_eval`) and Week 4
(`run_independent_sweep`, `counterfactual_effect`) stages already used
for the single-backbone run, so results are directly comparable.

Note the two "research-level" extra dependencies this needs beyond
EfficientNet-B4: `timm` for Xception/ViT/CLIP-RN50 (see requirements.txt).

Usage:
    python -m src.experiments.backbone_ablation \\
        --data_root ./data/ffplus_processed \\
        --backbones efficientnet_b4 xception vit_small_patch16_224 \\
        --epochs 20
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.data.datasets import AnnotatedFaceDataset, CounterfactualDataset, collate_counterfactual_batch
from src.detector.baseline_model import BaselineDeepfakeDetector
from src.detector.train_baseline import train_baseline
from src.detector.subgroup_eval import run_subgroup_eval
from src.disentangle.counterfactual_eval import counterfactual_effect, aggregate_by_identity
from src.utils.seed import set_seed

DEFAULT_BACKBONES = [
    "efficientnet_b4",   # locked primary detector
    "xception",
    "vit_small_patch16_224",
    "resnet50.clip_openai",
]


def run_backbone_ablation(
    data_root: str,
    backbones: list[str] = DEFAULT_BACKBONES,
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """For each backbone: train the baseline, run the Week 3 subgroup gap
    table, and run the Week 4 counterfactual causal test. Returns a dict
    keyed by backbone name so results.json is directly diffable across
    architectures."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = AnnotatedFaceDataset(data_root, split="train")
    val_ds = AnnotatedFaceDataset(data_root, split="val")
    test_ds = AnnotatedFaceDataset(data_root, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4)

    all_face_ids = train_ds.df["face_id"].tolist() + val_ds.df["face_id"].tolist()
    cf_ds = CounterfactualDataset(data_root, face_ids=all_face_ids)
    cf_loader = DataLoader(cf_ds, batch_size=batch_size, collate_fn=collate_counterfactual_batch)

    results = {}
    for name in backbones:
        print(f"\n{'=' * 40}\nBackbone: {name}\n{'=' * 40}")
        set_seed(seed)

        model = BaselineDeepfakeDetector(backbone_name=name)
        model = train_baseline(model, train_loader, val_loader, epochs=epochs, device=device)

        subgroup_results = run_subgroup_eval(model, test_loader, device=device)

        import numpy as np
        all_effects = {"delta_skin": [], "delta_illum": [], "interaction_residual": [], "face_ids": []}
        for cf_batch in cf_loader:
            effect = counterfactual_effect(model, cf_batch)
            for k in ("delta_skin", "delta_illum", "interaction_residual"):
                all_effects[k].append(effect[k])
            all_effects["face_ids"].extend(effect["face_ids"])
        for k in ("delta_skin", "delta_illum", "interaction_residual"):
            all_effects[k] = np.concatenate(all_effects[k])
        effect_by_identity = aggregate_by_identity(all_effects)

        results[name] = {
            "subgroup_gaps": {
                "by_skin_tone": subgroup_results["by_skin_tone"].get("__gap__", {}),
                "by_illumination": subgroup_results["by_illumination"].get("__gap__", {}),
            },
            "mean_abs_delta_skin": float(np.mean(np.abs(effect_by_identity["delta_skin"]))),
            "mean_abs_delta_illum": float(np.mean(np.abs(effect_by_identity["delta_illum"]))),
        }
        skin_gt_illum = results[name]["mean_abs_delta_skin"] > results[name]["mean_abs_delta_illum"]
        results[name]["skin_effect_larger"] = bool(skin_gt_illum)

    return results


def summarize_backbone_agreement(results: dict) -> str:
    """One-line summary: do all backbones agree on the direction of the
    skin-vs-illumination effect? This is the actual research question the
    ablation exists to answer."""
    directions = {name: r["skin_effect_larger"] for name, r in results.items()}
    if len(set(directions.values())) == 1:
        direction = "skin tone" if next(iter(directions.values())) else "illumination"
        return f"All {len(directions)} backbones agree: {direction} produces the larger counterfactual effect."
    agree = [n for n, v in directions.items() if v]
    disagree = [n for n, v in directions.items() if not v]
    return (
        f"Backbones DISAGREE on direction: skin-tone-larger={agree}, "
        f"illumination-larger={disagree}. The Delta_skin > Delta_illum finding is "
        f"NOT backbone-invariant and should not be reported as a general claim."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True)
    p.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Path to write JSON results (default: <data_root>/backbone_ablation.json)")
    return p


def main():
    args = build_arg_parser().parse_args()
    results = run_backbone_ablation(
        data_root=args.data_root,
        backbones=args.backbones,
        epochs=args.epochs,
        seed=args.seed,
    )
    output_path = args.output or f"{args.data_root.rstrip('/')}/backbone_ablation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{summarize_backbone_agreement(results)}")
    print(f"[backbone_ablation] results written to {output_path}")


if __name__ == "__main__":
    # summarize_backbone_agreement is pure and torch-free -- smoke test it
    # directly. The full run_backbone_ablation() path needs torch + timm +
    # real data.
    agree_case = {
        "efficientnet_b4": {"skin_effect_larger": True},
        "xception": {"skin_effect_larger": True},
    }
    disagree_case = {
        "efficientnet_b4": {"skin_effect_larger": True},
        "vit_small_patch16_224": {"skin_effect_larger": False},
    }
    msg_agree = summarize_backbone_agreement(agree_case)
    msg_disagree = summarize_backbone_agreement(disagree_case)
    assert "agree" in msg_agree.lower() and "DISAGREE" not in msg_agree
    assert "DISAGREE" in msg_disagree
    print("[smoke test]", msg_agree)
    print("[smoke test]", msg_disagree)
    print("[smoke test] PASSED")
