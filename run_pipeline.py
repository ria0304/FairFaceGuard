"""End-to-end orchestrator for the 5-week roadmap.

This wires every phase together in order. It's meant to be adapted, not run
blindly -- fill in your actual data_root and adjust batch sizes / epochs for
your hardware. Each stage is also independently runnable/testable as its
own module (see the __main__ blocks in src/annotation, src/augmentation).

Usage:
    python run_pipeline.py --data_root /path/to/data --stage all
    python run_pipeline.py --data_root /path/to/data --stage annotate
    python run_pipeline.py --data_root /path/to/data --stage augment
    python run_pipeline.py --data_root /path/to/data --stage baseline
    python run_pipeline.py --data_root /path/to/data --stage disentangle
    python run_pipeline.py --data_root /path/to/data --stage report
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.annotation.ita_fitzpatrick import annotate_face
from src.augmentation.counterfactual import generate_counterfactual_set
from src.data.datasets import AnnotatedFaceDataset, CounterfactualDataset, collate_counterfactual_batch
from src.detector.baseline_model import BaselineDeepfakeDetector
from src.detector.train_baseline import train_baseline
from src.detector.subgroup_eval import run_subgroup_eval, print_gap_report
from src.disentangle.train import run_independent_sweep
from src.disentangle.probing import run_leakage_report
from src.disentangle.counterfactual_eval import counterfactual_effect, aggregate_by_identity
from src.reports.gap_tables import build_final_report, save_report
from src.utils.seed import set_seed


def stage_annotate(data_root: str) -> None:
    """Week 1: run ITA/Fitzpatrick annotation over every frame in frames/."""
    frame_paths = sorted(glob.glob(os.path.join(data_root, "frames", "*.png")))
    rows = []
    for path in frame_paths:
        face_id = os.path.splitext(os.path.basename(path))[0]
        image = cv2.imread(path)
        result = annotate_face(image, face_id)
        row = result.to_dict()
        illum = row.pop("illuminant_estimate")
        row["illuminant_estimate_b"], row["illuminant_estimate_g"], row["illuminant_estimate_r"] = illum
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(data_root, "annotations.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} annotations to {out_path}")


def stage_augment(data_root: str) -> None:
    """Week 2: generate counterfactual sets for every annotated frame."""
    annotations = pd.read_csv(os.path.join(data_root, "annotations.csv"))
    out_root = os.path.join(data_root, "counterfactuals")
    for _, row in annotations.iterrows():
        face_id = row["face_id"]
        img_path = os.path.join(data_root, "frames", f"{face_id}.png")
        image = cv2.imread(img_path)
        cf = generate_counterfactual_set(image, face_id)

        face_dir = os.path.join(out_root, face_id)
        os.makedirs(face_dir, exist_ok=True)
        cv2.imwrite(os.path.join(face_dir, "original.png"), cf.original)
        cv2.imwrite(os.path.join(face_dir, "skin_only.png"), cf.skin_only)
        cv2.imwrite(os.path.join(face_dir, "illum_only.png"), cf.illum_only)
        cv2.imwrite(os.path.join(face_dir, "both.png"), cf.both)

        if not cf.validation["ita_manipulation_check_pass"]:
            print(f"  [WARN] {face_id}: skin-tone counterfactual failed manipulation check: {cf.validation}")

    print(f"Wrote counterfactual sets to {out_root}")


def stage_baseline(data_root: str, epochs: int = 20) -> BaselineDeepfakeDetector:
    """Week 3: train baseline detector, then produce subgroup gap tables."""
    train_ds = AnnotatedFaceDataset(data_root, split="train")
    val_ds = AnnotatedFaceDataset(data_root, split="val")
    test_ds = AnnotatedFaceDataset(data_root, split="test")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=4)

    model = BaselineDeepfakeDetector()
    model = train_baseline(model, train_loader, val_loader, epochs=epochs)
    torch.save(model.state_dict(), os.path.join(data_root, "baseline_model.pt"))

    results = run_subgroup_eval(model, test_loader)
    print_gap_report(results)
    return model, results


def stage_disentangle(data_root: str, baseline_model: BaselineDeepfakeDetector, epochs_per_run: int = 15) -> dict:
    """Week 4: adversarial sweep + frozen-model probing + counterfactual causal test."""
    train_ds = AnnotatedFaceDataset(data_root, split="train")
    val_ds = AnnotatedFaceDataset(data_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)

    sweep_results = run_independent_sweep(
        train_loader, val_loader,
        baseline_state_dict=baseline_model.backbone.state_dict(),
        epochs_per_run=epochs_per_run,
    )

    leakage_results = run_leakage_report(baseline_model, train_loader, val_loader)

    all_face_ids = train_ds.df["face_id"].tolist() + val_ds.df["face_id"].tolist()
    cf_ds = CounterfactualDataset(data_root, face_ids=all_face_ids)
    cf_loader = DataLoader(cf_ds, batch_size=32, collate_fn=collate_counterfactual_batch)

    all_effects = {"delta_skin": [], "delta_illum": [], "interaction_residual": [], "face_ids": []}
    for cf_batch in cf_loader:
        effect = counterfactual_effect(baseline_model, cf_batch)
        for k in ("delta_skin", "delta_illum", "interaction_residual"):
            all_effects[k].append(effect[k])
        all_effects["face_ids"].extend(effect["face_ids"])

    import numpy as np
    for k in ("delta_skin", "delta_illum", "interaction_residual"):
        all_effects[k] = np.concatenate(all_effects[k])

    effect_by_identity = aggregate_by_identity(all_effects)
    return {"sweep_results": sweep_results, "leakage_results": leakage_results, "effect_by_identity": effect_by_identity}


def stage_report(data_root: str, subgroup_results: dict, week4_results: dict) -> None:
    """Week 5: aggregate everything into the final paper-ready report."""
    report = build_final_report(
        subgroup_eval_results=subgroup_results,
        sweep_results=week4_results["sweep_results"],
        leakage_results=week4_results["leakage_results"],
        counterfactual_effect_by_identity=week4_results["effect_by_identity"],
    )
    save_report(report, os.path.join(data_root, "final_report.json"))
    print(report["headline_conclusion"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--stage", default="all", choices=["all", "annotate", "augment", "baseline", "disentangle", "report"])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    set_seed(42)

    if args.stage in ("all", "annotate"):
        stage_annotate(args.data_root)
    if args.stage in ("all", "augment"):
        stage_augment(args.data_root)

    baseline_model, subgroup_results = (None, None)
    if args.stage in ("all", "baseline"):
        baseline_model, subgroup_results = stage_baseline(args.data_root, epochs=args.epochs)

    week4_results = None
    if args.stage in ("all", "disentangle"):
        if baseline_model is None:
            baseline_model = BaselineDeepfakeDetector()
            baseline_model.load_state_dict(torch.load(os.path.join(args.data_root, "baseline_model.pt")))
        week4_results = stage_disentangle(args.data_root, baseline_model)

    if args.stage in ("all", "report"):
        stage_report(args.data_root, subgroup_results, week4_results)
