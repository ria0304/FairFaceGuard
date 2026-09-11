"""Week 3 -- subgroup accuracy/EER gap tables.

Evaluates the trained baseline on the annotated (Week 1) test set and
produces the Fitzpatrick-bin x illumination-bin gap table this phase is
meant to deliver. This is purely descriptive -- it establishes THAT a gap
exists; Week 4 establishes WHY.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.detector.baseline_model import BaselineDeepfakeDetector
from src.utils.metrics import subgroup_metric_table


FITZPATRICK_NAMES = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}
ILLUM_BIN_NAMES = {0: "very_dark", 1: "dark", 2: "mid", 3: "bright", 4: "very_bright"}


@torch.no_grad()
def run_subgroup_eval(
    model: BaselineDeepfakeDetector,
    test_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """test_loader must yield {'image', 'fake_label', 'skin_bin', 'illum_bin'}.
    Returns a dict with three tables: by skin bin, by illumination bin, and
    the 2D cross table (skin x illum -> EER), plus overall max-min gaps."""
    model.eval().to(device)

    y_true, y_score, skin_bins, illum_bins = [], [], [], []
    for batch in test_loader:
        x = batch["image"].to(device)
        p_fake = model.predict_proba(x).cpu().numpy()
        y_true.append(batch["fake_label"].numpy())
        y_score.append(p_fake)
        skin_bins.append(batch["skin_bin"].numpy())
        illum_bins.append(batch["illum_bin"].numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    skin_bins = np.concatenate(skin_bins)
    illum_bins = np.concatenate(illum_bins)

    by_skin = subgroup_metric_table(y_true, y_score, skin_bins, FITZPATRICK_NAMES)
    by_illum = subgroup_metric_table(y_true, y_score, illum_bins, ILLUM_BIN_NAMES)
    cross = _cross_eer_table(y_true, y_score, skin_bins, illum_bins)

    return {"by_skin_tone": by_skin, "by_illumination": by_illum, "cross_table_eer": cross}


def _cross_eer_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    skin_bins: np.ndarray,
    illum_bins: np.ndarray,
) -> dict:
    from src.utils.metrics import equal_error_rate

    table = {}
    for s in np.unique(skin_bins):
        row = {}
        for i in np.unique(illum_bins):
            mask = (skin_bins == s) & (illum_bins == i)
            if mask.sum() < 5 or len(np.unique(y_true[mask])) < 2:
                row[ILLUM_BIN_NAMES.get(int(i), str(i))] = None
                continue
            row[ILLUM_BIN_NAMES.get(int(i), str(i))] = round(
                equal_error_rate(y_true[mask], y_score[mask]), 4
            )
        table[FITZPATRICK_NAMES.get(int(s), str(s))] = row
    return table


def print_gap_report(results: dict) -> None:
    print("=== Subgroup gap report (Week 3) ===")
    for axis in ("by_skin_tone", "by_illumination"):
        print(f"\n-- {axis} --")
        table = results[axis]
        gap = table.get("__gap__", {})
        for name, m in table.items():
            if name == "__gap__":
                continue
            print(f"  {name:12s}  n={m['n']:5d}  acc={m['accuracy']:.4f}  auc={m['auc']:.4f}  eer={m['eer']:.4f}")
        print(f"  GAP (max-min): accuracy={gap.get('accuracy', float('nan')):.4f}  "
              f"auc={gap.get('auc', float('nan')):.4f}  eer={gap.get('eer', float('nan')):.4f}")

    print("\n-- cross table (EER), rows=Fitzpatrick, cols=illumination --")
    for row_name, row in results["cross_table_eer"].items():
        cells = "  ".join(f"{k}={v}" for k, v in row.items())
        print(f"  {row_name}: {cells}")
