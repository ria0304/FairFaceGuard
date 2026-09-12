"""Dataset-level rollup of the Week-2 counterfactual validation gate
(src/augmentation/counterfactual.py::_validate_counterfactuals).

Every CounterfactualSet already carries its own per-face `accepted` flag
and `reject_reasons`. This module aggregates those across a whole dataset
run so the rejection/exclusion rate -- required reporting for the
counterfactual validation gate -- is produced automatically instead of
needing to be tallied by hand from log lines.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

import pandas as pd

from src.augmentation.counterfactual import CounterfactualSet

logger = logging.getLogger(__name__)


class ValidationTracker:
    """Accumulate per-face validation outcomes while a dataset is being
    augmented, then produce a summary + a row-per-face CSV of flags."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, cf: CounterfactualSet) -> None:
        v = cf.validation
        self._rows.append({
            "face_id": cf.face_id,
            "accepted": v["accepted"],
            "skin_only_accepted": v["skin_only_accepted"],
            "illum_only_accepted": v["illum_only_accepted"],
            "both_accepted": v["both_accepted"],
            "reject_reasons": "; ".join(v["reject_reasons"]),
            "measured_ita_delta_skin_only": v["measured_ita_delta_skin_only"],
            "residual_illuminant_coupling_in_skin_shift": v["residual_illuminant_coupling_in_skin_shift"],
            "residual_ita_coupling_in_illum_shift": v["residual_ita_coupling_in_illum_shift"],
            "measured_ita_delta_both": v["measured_ita_delta_both"],
            "both_illum_shift_magnitude": v["both_illum_shift_magnitude"],
        })

    def summary(self) -> dict:
        n = len(self._rows)
        if n == 0:
            return {"n_total": 0, "n_accepted": 0, "n_rejected": 0, "rejection_rate": float("nan")}

        n_accepted = sum(1 for r in self._rows if r["accepted"])
        n_rejected = n - n_accepted

        reason_counts: Counter[str] = Counter()
        for r in self._rows:
            for reason in r["reject_reasons"].split("; "):
                if reason:
                    reason_counts[reason] += 1

        return {
            "n_total": n,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "rejection_rate": round(n_rejected / n, 4),
            "exclusion_rate": round(n_rejected / n, 4),  # alias, same quantity
            "n_skin_only_rejected": sum(1 for r in self._rows if not r["skin_only_accepted"]),
            "n_illum_only_rejected": sum(1 for r in self._rows if not r["illum_only_accepted"]),
            "n_both_rejected": sum(1 for r in self._rows if not r["both_accepted"]),
            "rejection_reason_counts": dict(reason_counts),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def save(self, csv_path: str, summary_json_path: str | None = None) -> dict:
        """Writes the per-face flag table to `csv_path` and (optionally) the
        aggregate summary to `summary_json_path`. Returns the summary dict."""
        self.to_dataframe().to_csv(csv_path, index=False)
        logger.info("Per-face validation flags (CSV) written to %s", csv_path)
        summary = self.summary()
        if summary_json_path:
            with open(summary_json_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Validation summary (JSON) written to %s", summary_json_path)
        return summary


def print_validation_summary(summary: dict) -> None:
    logger.info("=== Counterfactual validation gate summary ===")
    logger.info("  total samples:      %d", summary["n_total"])
    logger.info("  accepted:           %d", summary["n_accepted"])
    logger.info("  rejected:           %d", summary["n_rejected"])
    if summary["n_total"]:
        logger.info("  rejection rate:     %.2f%%", summary["rejection_rate"] * 100)
    else:
        logger.info("  rejection rate:     n/a")
    logger.info("    - failed skin_only check:  %d", summary.get("n_skin_only_rejected", 0))
    logger.info("    - failed illum_only check: %d", summary.get("n_illum_only_rejected", 0))
    logger.info("    - failed both check:       %d", summary.get("n_both_rejected", 0))
    if summary.get("rejection_reason_counts"):
        logger.info("  reasons:")
        for reason, count in summary["rejection_reason_counts"].items():
            logger.info("    - %s: %d", reason, count)
