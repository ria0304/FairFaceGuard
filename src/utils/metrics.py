="""Shared evaluation metrics used across the baseline (Week 3), disentanglement
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
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    results = {}
    for g in np.unique(group_labels):
        mask = group_labels == g
        if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
            continue  # not enough data / only one class present
        y_t, y_s = y_true[mask], y_score[mask]
        y_pred = (y_s >= 0.5).astype(int)
        name = group_names.get(int(g), str(g)) if group_names else str(g)
        
        # Compute all metrics
        acc = float(accuracy_score(y_t, y_pred))
        auc = float(roc_auc_score(y_t, y_s))
        eer_val = equal_error_rate(y_t, y_s)
        prec = float(precision_score(y_t, y_pred, zero_division=np.nan))
        rec = float(recall_score(y_t, y_pred, zero_division=np.nan))
        f1 = float(f1_score(y_t, y_pred, zero_division=np.nan))
        
        # FPR and FNR
        fp = np.sum((y_pred == 1) & (y_t == 0))
        tn = np.sum((y_pred == 0) & (y_t == 0))
        fn = np.sum((y_pred == 0) & (y_t == 1))
        tp = np.sum((y_pred == 1) & (y_t == 1))
        
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else np.nan
        
        results[name] = {
            "n": int(mask.sum()),
            "accuracy": acc,
            "auc": auc,
            "eer": eer_val,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
        }

    if results:
        for metric in ("accuracy", "auc", "eer", "fpr", "fnr"):
            vals = [v[metric] for v in results.values() if not np.isnan(v.get(metric, np.nan))]
            if len(vals) >= 2:
                results.setdefault("__gap__", {})[metric] = max(vals) - min(vals)

    return results


def compute_eer_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Score threshold at the equal-error-rate operating point -- used by
    compute_acer as the default decision threshold when none is supplied."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(thresholds[idx])


def compute_acer(y_true: np.ndarray, y_score: np.ndarray, threshold: float | None = None) -> float:
    """Average Classification Error Rate = (APCER + BPCER) / 2, the standard
    reporting metric in face anti-spoofing / deepfake-detection literature
    (ISO/IEC 30107-3). APCER = fake classified as real (attack presentation
    error), BPCER = real classified as fake (bona fide presentation error).
    Uses the EER threshold by default so ACER reduces to a close cousin of
    EER at threshold=0.5 is not assumed."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if threshold is None:
        threshold = compute_eer_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    fake_mask = y_true == 1
    real_mask = y_true == 0
    apcer = float(np.mean(y_pred[fake_mask] == 0)) if fake_mask.any() else float("nan")
    bpcer = float(np.mean(y_pred[real_mask] == 1)) if real_mask.any() else float("nan")
    return (apcer + bpcer) / 2.0


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve. More informative than AUC
    under class imbalance (e.g. a manipulation-method subset with far more
    fake than real frames), which subgroup slices of FF++/Celeb-DF often
    have."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_score))


def subgroup_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    skin_bins: np.ndarray,
    illum_bins: np.ndarray,
    skin_names: dict[int, str] | None = None,
    illum_names: dict[int, str] | None = None,
    min_n: int = 5,
):
    """2D accuracy matrix, Fitzpatrick skin bin x illumination bin, as a
    pandas DataFrame suitable for a heatmap in the paper. Cells with fewer
    than `min_n` samples are left as NaN rather than an unreliable point
    estimate."""
    import pandas as pd

    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "correct": (np.asarray(y_true) == np.asarray(y_pred)).astype(float),
        "skin": np.asarray(skin_bins),
        "illum": np.asarray(illum_bins),
    })

    def _agg(group):
        return group["correct"].mean() if len(group) >= min_n else np.nan

    table = df.groupby(["skin", "illum"]).apply(_agg).unstack("illum")
    if skin_names:
        table = table.rename(index=skin_names)
    if illum_names:
        table = table.rename(columns=illum_names)
    return table


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    cluster_ids: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a 1D array of values.

    IMPORTANT: If cluster_ids is provided, resamples at the cluster level
    (e.g., identity/video) rather than individual observations. This is
    essential for correlated data like multiple frames per subject.
    
    Args:
        values: Array of values (should be aggregated at identity level if
            data has clustering).
        n_boot: Number of bootstrap iterations.
        alpha: Significance level for CI (default 0.05 gives 95% CI).
        seed: Random seed for reproducibility.
        cluster_ids: Optional array of cluster identifiers. If provided,
            bootstrap resamples clusters rather than individual values.

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    
    if cluster_ids is not None and len(cluster_ids) == n:
        # Cluster-aware bootstrap
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        
        # Compute cluster means
        cluster_means = []
        for cid in unique_clusters:
            mask = cluster_ids == cid
            cluster_means.append(values[mask].mean())
        cluster_means = np.array(cluster_means)
        
        # Resample clusters
        idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        boot_means = cluster_means[idx].mean(axis=1)
    else:
        # Standard bootstrap
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
    paired: bool = False,
) -> float:
    """Two-sided permutation test p-value for mean(a) != mean(b).
    
    Used in Week 5 to test whether Delta_skin significantly differs from
    Delta_illum (not just whether each differs from zero).
    
    Args:
        a: First array of values.
        b: Second array of values.
        n_perm: Number of permutations.
        seed: Random seed.
        paired: If True, performs a paired permutation test (sign-flip test)
            appropriate for measurements on the same samples. If False, uses
            standard unpaired permutation test.
    
    Returns:
        Two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    
    if paired:
        # Paired permutation test (sign-flip test)
        # For paired observations, we test if the mean difference is zero
        # by randomly flipping signs of the differences
        if len(a) != len(b):
            raise ValueError("For paired test, arrays must have same length")
        
        diff = a - b
        observed = abs(diff.mean())
        
        count = 0
        for _ in range(n_perm):
            # Randomly flip signs
            signs = rng.choice([-1, 1], size=len(diff))
            permuted_diff = diff * signs
            permuted_stat = abs(permuted_diff.mean())
            if permuted_stat >= observed:
                count += 1
    else:
        # Unpaired permutation test
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


def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 5000,
    seed: int = 0,
) -> float:
    """Paired permutation test (sign-flip test) for comparing two related samples.
    
    This is the appropriate test when a and b are measurements on the SAME
    subjects/samples (e.g., Delta_skin and Delta_illum on the same images).
    
    Args:
        a: First array of paired measurements.
        b: Second array of paired measurements.
        n_perm: Number of permutations.
        seed: Random seed.
    
    Returns:
        Two-sided p-value.
    """
    return permutation_test_diff(a, b, n_perm=n_perm, seed=seed, paired=True)


def wilcoxon_signed_rank_test(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples.
    
    Non-parametric alternative to paired t-test. Tests whether the median
    difference between pairs is zero.
    
    Args:
        a: First array of paired measurements.
        b: Second array of paired measurements.
    
    Returns:
        Tuple of (test_statistic, p_value).
    """
    from scipy.stats import wilcoxon
    
    result = wilcoxon(a, b)
    return float(result.statistic), float(result.pvalue)


def cohen_kappa(y1: np.ndarray, y2: np.ndarray, weights: str | None = None) -> float:
    """Cohen's kappa for inter-rater agreement.
    
    Args:
        y1: First set of labels.
        y2: Second set of labels.
        weights: Weighting scheme ('linear', 'quadratic', or None for unweighted).
    
    Returns:
        Kappa coefficient.
    """
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y1, y2, weights=weights))


def effect_size_cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for difference between two means.
    
    Args:
        a: First sample.
        b: Second sample.
    
    Returns:
        Cohen's d value.
    """
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((mean_a - mean_b) / pooled_std)


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values to correct.
        alpha: Significance level.
    
    Returns:
        Tuple of (corrected_p_values, significance_flags).
    """
    m = len(p_values)
    corrected = [min(p * m, 1.0) for p in p_values]
    significant = [p < alpha for p in corrected]
    return corrected, significant


def benjamini_hochberg_correction(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg procedure for controlling false discovery rate.
    
    Args:
        p_values: List of p-values to correct.
        alpha: Significance level.
    
    Returns:
        Tuple of (adjusted_p_values, significance_flags).
    """
    import numpy as np
    
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Calculate adjusted p-values
    adjusted = np.zeros(m)
    for i in range(m - 1, -1, -1):
        adjusted[i] = min(1.0, sorted_p[i] * m / (i + 1))
        if i < m - 1:
            adjusted[i] = min(adjusted[i], adjusted[i + 1])
    
    # Restore original order
    corrected = np.zeros(m)
    corrected[sorted_indices] = adjusted
    
    significant = corrected < alpha
    return corrected.tolist(), significant.tolist()
