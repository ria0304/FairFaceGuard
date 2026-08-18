"""Shared evaluation metrics used across the baseline (Week 3), disentanglement
(Week 4), and reporting (Week 5) stages."""

from __future__ import annotations

import numpy as np


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Equal Error Rate: the point where false positive rate
    equals false negative rate on the ROC curve.

    y_true:  binary labels, 1 = fake, 0 = real
    y_score: predicted P(fake)
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # index where fpr and fnr cross (fpr - fnr changes sign)
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def subgroup_metric_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_labels: np.ndarray,
    group_names: dict[int, str] | None = None,
) -> dict:
    """Compute accuracy, AUC, EER per group. Returns a dict keyed by group id,
    plus the overall max-min gap for each metric (the "subgroup gap" reported
    in Week 3 / Week 5 tables)."""
    from sklearn.metrics import roc_auc_score, accuracy_score

    results = {}
    for g in np.unique(group_labels):
        mask = group_labels == g
        if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
            continue  # not enough data / only one class present
        y_t, y_s = y_true[mask], y_score[mask]
        name = group_names.get(int(g), str(g)) if group_names else str(g)
        results[name] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_t, y_s >= 0.5)),
            "auc": float(roc_auc_score(y_t, y_s)),
            "eer": equal_error_rate(y_t, y_s),
        }

    if results:
        for metric in ("accuracy", "auc", "eer"):
            vals = [v[metric] for v in results.values()]
            results.setdefault("__gap__", {})[metric] = max(vals) - min(vals)

    return results


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a 1D array of per-identity effect sizes.

    IMPORTANT: pass values already aggregated at the IDENTITY level (e.g. one
    mean-delta per subject), not per-frame, to avoid inflated confidence from
    correlated frames of the same face -- see methodology Section 5/6.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def permutation_test_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 5000,
    seed: int = 0,
) -> float:
    """Two-sided permutation test p-value for mean(a) != mean(b).
    Used in Week 5 to test whether Delta_skin significantly differs from
    Delta_illum (not just whether each differs from zero)."""
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(pooled[:n_a].mean() - pooled[n_a:].mean())
        if diff >= observed:
            count += 1
    return count / n_perm
