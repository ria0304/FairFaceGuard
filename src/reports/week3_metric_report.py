"""Week 3 -- single-command metric report.

Runs the frozen baseline detector once over the test set and produces every
metric this phase is required to deliver, written to BOTH a JSON report
(nested, for programmatic use) and a flat CSV (one row per scope/group, for
opening in a spreadsheet):

    overall:            ROC-AUC, Accuracy, Precision, Recall/TPR, F1, FPR, FNR, EER
    by_skin_tone:        the same metrics per Fitzpatrick bin
    by_illumination:     the same metrics per illumination bin
    gaps:                delta_AUC, TPR gap, FPR gap, EER (max-min across
                         skin-tone groups) -- plus the same gaps by illumination

This is the one script `python -m src.reports.week3_metric_report` (or
`build_week3_report(...)` called from run_pipeline.py) that automatically
produces the whole Week-3 deliverable -- no metric here needs a second,
separately-run script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.detector.baseline_model import BaselineDeepfakeDetector
from src.detector.subgroup_eval import FITZPATRICK_NAMES, ILLUM_BIN_NAMES, run_subgroup_eval
from src.utils.metrics import equal_error_rate

logger = logging.getLogger(__name__)


REQUIRED_GAP_METRICS = ("auc", "recall", "fpr", "fnr", "eer", "accuracy")
# Human-readable aliases for the report so "recall gap" also reads as "TPR gap".
GAP_METRIC_DISPLAY_NAMES = {
    "auc": "delta_auc",
    "recall": "tpr_gap",
    "fpr": "fpr_gap",
    "fnr": "fnr_gap",
    "eer": "eer_gap",
    "accuracy": "accuracy_gap",
}


@torch.no_grad()
def _overall_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    )

    y_pred = (y_score >= 0.5).astype(int)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan")

    return {
        "n": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=np.nan)),
        "recall_tpr": float(recall_score(y_true, y_pred, zero_division=np.nan)),
        "f1": float(f1_score(y_true, y_pred, zero_division=np.nan)),
        "fpr": fpr,
        "fnr": fnr,
        "eer": equal_error_rate(y_true, y_score),
    }


def _gaps_from_subgroup_table(table: dict) -> dict:
    """Pull the max-min gaps that subgroup_metric_table already computed
    (src/utils/metrics.py) and relabel them with the names this report
    is required to surface (delta_auc, tpr_gap, fpr_gap, ...)."""
    raw_gaps = table.get("__gap__", {})
    return {
        GAP_METRIC_DISPLAY_NAMES[m]: raw_gaps.get(m, float("nan"))
        for m in REQUIRED_GAP_METRICS
    }


def _per_group_summary(table: dict) -> dict:
    """Drop the internal __gap__ key and round for a readable per-group table."""
    return {name: m for name, m in table.items() if name != "__gap__"}


@torch.no_grad()
def build_week3_report(
    model: BaselineDeepfakeDetector,
    test_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Single entry point: runs the model once, then assembles every
    required Week-3 metric (overall + per-skin-tone + per-illumination +
    gaps) into one report dict."""
    model.eval().to(device)

    y_true, y_score = [], []
    for batch in test_loader:
        x = batch["image"].to(device)
        p_fake = model.predict_proba(x).cpu().numpy()
        y_true.append(batch["fake_label"].numpy())
        y_score.append(p_fake)
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    subgroup_results = run_subgroup_eval(model, test_loader, device=device)
    by_skin = subgroup_results["by_skin_tone"]
    by_illum = subgroup_results["by_illumination"]

    report = {
        "overall": _overall_metrics(y_true, y_score),
        "by_skin_tone": _per_group_summary(by_skin),
        "by_illumination": _per_group_summary(by_illum),
        "gaps_by_skin_tone": _gaps_from_subgroup_table(by_skin),
        "gaps_by_illumination": _gaps_from_subgroup_table(by_illum),
        "eer_by_skin_tone_group": {
            name: m["eer"] for name, m in _per_group_summary(by_skin).items()
        },
        "cross_table_eer_skin_x_illum": subgroup_results["cross_table_eer"],
    }
    return report


def save_report(report: dict, path: str) -> None:
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_default)
    logger.info("Week-3 metric report (JSON) written to %s", path)


def report_to_dataframe(report: dict) -> pd.DataFrame:
    """Flattens the nested report into one row per (scope, group) so it can
    be opened directly in a spreadsheet / loaded with pandas. `scope` is one
    of 'overall', 'by_skin_tone', 'by_illumination'; the gap rows carry
    scope='gap' with the gap metrics in the same metric columns."""
    rows = []

    overall = dict(report["overall"])
    overall_row = {"scope": "overall", "group": "all"}
    overall_row.update(overall)
    rows.append(overall_row)

    for axis_key, axis_label in (("by_skin_tone", "skin_tone"), ("by_illumination", "illumination")):
        for group_name, m in report[axis_key].items():
            row = {"scope": axis_label, "group": group_name}
            row.update(m)
            rows.append(row)

        gap_key = "gaps_by_skin_tone" if axis_key == "by_skin_tone" else "gaps_by_illumination"
        gap_row = {"scope": f"{axis_label}_gap", "group": "max_minus_min"}
        gap_row.update(report[gap_key])
        rows.append(gap_row)

    return pd.DataFrame(rows)


def save_report_csv(report: dict, path: str) -> None:
    df = report_to_dataframe(report)
    df.to_csv(path, index=False)
    logger.info("Week-3 metric report (CSV) written to %s", path)


def print_report(report: dict) -> None:
    logger.info("=== Week-3 metric report ===")
    logger.info("-- overall --")
    for k, v in report["overall"].items():
        logger.info("  %-12s %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    for axis_key, axis_label in (
        ("by_skin_tone", "by skin tone"),
        ("by_illumination", "by illumination"),
    ):
        logger.info("-- %s --", axis_label)
        for name, m in report[axis_key].items():
            logger.info(
                "  %-12s n=%5d  auc=%.4f  acc=%.4f  prec=%.4f  recall=%.4f  f1=%.4f  fpr=%.4f  fnr=%.4f  eer=%.4f",
                name, m["n"], m["auc"], m["accuracy"], m["precision"], m["recall"], m["f1"], m["fpr"], m["fnr"], m["eer"],
            )
        gap_key = "gaps_by_skin_tone" if axis_key == "by_skin_tone" else "gaps_by_illumination"
        gaps = report[gap_key]
        logger.info(
            "  GAPS: delta_auc=%.4f  tpr_gap=%.4f  fpr_gap=%.4f  fnr_gap=%.4f  eer_gap=%.4f  accuracy_gap=%.4f",
            gaps["delta_auc"], gaps["tpr_gap"], gaps["fpr_gap"], gaps["fnr_gap"], gaps["eer_gap"], gaps["accuracy_gap"],
        )


if __name__ == "__main__":
    from src.data.datasets import AnnotatedFaceDataset

    parser = argparse.ArgumentParser(description="Week-3 single-command metric report")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to trained baseline_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_json", default=None, help="Defaults to <data_root>/week3_metric_report.json")
    parser.add_argument("--out_csv", default=None, help="Defaults to <data_root>/week3_metric_report.csv")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineDeepfakeDetector()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_ds = AnnotatedFaceDataset(args.data_root, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=4)

    report = build_week3_report(model, test_loader, device=device)
    print_report(report)
    save_report(report, args.out_json or os.path.join(args.data_root, "week3_metric_report.json"))
    save_report_csv(report, args.out_csv or os.path.join(args.data_root, "week3_metric_report.csv"))
