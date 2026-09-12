"""Week 3/4/5 -- automatically generate the required results figures.

One call (`generate_all_required_figures`) produces every figure this
deliverable lists:

    - ROC curves by skin-tone group
    - AUC by skin-tone group
    - TPR/FPR by group
    - EER by group
    - Original vs skin-only
    - Original vs lighting-only
    - Original vs combined
    - Performance-drop comparison

Each figure is also exposed as its own function so a single plot can be
regenerated (e.g. for a paper revision) without rerunning everything.
Every function takes already-computed arrays/report dicts -- the same
shapes produced by src/detector/subgroup_eval.py,
src/reports/week3_metric_report.py, and
src/reports/disentanglement_calculator.py -- so this module never runs the
model itself.
"""

from __future__ import annotations

import os

import numpy as np


def _new_fig(figsize=(6, 5)):
    import matplotlib

    matplotlib.use("Agg")  # headless: never require a display
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    return plt, fig, ax


def _savefig(plt, fig, save_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# =============================================================================
# 1. ROC curves by skin-tone (or illumination) group
# =============================================================================
def plot_roc_curves_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_labels: np.ndarray,
    group_names: dict[int, str] | None,
    save_path: str,
    title: str = "ROC curves by skin-tone group",
) -> str:
    from sklearn.metrics import roc_curve, auc as sk_auc

    plt, fig, ax = _new_fig()
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    group_labels = np.asarray(group_labels)

    for g in np.unique(group_labels):
        mask = group_labels == g
        if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
        name = group_names.get(int(g), str(g)) if group_names else str(g)
        ax.plot(fpr, tpr, label=f"{name} (AUC={sk_auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    return _savefig(plt, fig, save_path)


# =============================================================================
# 2/3/4. Bar chart of one metric across groups (AUC by group, EER by group,
# and, with metric="recall"/"fpr" plotted together, TPR/FPR by group)
# =============================================================================
def plot_metric_by_group(
    per_group_table: dict,
    metric: str,
    save_path: str,
    title: str | None = None,
    ylabel: str | None = None,
) -> str:
    """per_group_table: e.g. results["by_skin_tone"] from subgroup_eval.py
    (a name -> {n, accuracy, auc, eer, fpr, recall, ...} dict). The
    internal "__gap__" key, if present, is skipped."""
    plt, fig, ax = _new_fig()

    names = [n for n in per_group_table.keys() if n != "__gap__"]
    values = [per_group_table[n][metric] for n in names]

    bars = ax.bar(names, values, color="#1f6fb2")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(ylabel or metric.upper())
    ax.set_title(title or f"{metric.upper()} by group")
    ax.set_xlabel("Group")
    return _savefig(plt, fig, save_path)


def plot_tpr_fpr_by_group(
    per_group_table: dict,
    save_path: str,
    title: str = "TPR / FPR by group",
) -> str:
    plt, fig, ax = _new_fig()

    names = [n for n in per_group_table.keys() if n != "__gap__"]
    tpr = [per_group_table[n]["recall"] for n in names]  # recall == TPR
    fpr = [per_group_table[n]["fpr"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, tpr, width, label="TPR", color="#2a9d8f")
    ax.bar(x + width / 2, fpr, width, label="FPR", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    return _savefig(plt, fig, save_path)


def plot_eer_by_group(per_group_table: dict, save_path: str, title: str = "EER by group") -> str:
    return plot_metric_by_group(per_group_table, "eer", save_path, title=title, ylabel="EER")


def plot_auc_by_group(per_group_table: dict, save_path: str, title: str = "AUC by group") -> str:
    return plot_metric_by_group(per_group_table, "auc", save_path, title=title, ylabel="AUC")


# =============================================================================
# 5/6/7. Original vs skin-only / lighting-only / combined -- one ROC overlay
# per requested comparison, using the counterfactual-condition scores from
# src/reports/disentanglement_calculator.py.
# =============================================================================
def plot_original_vs_condition_roc(
    y_true: np.ndarray,
    score_original: np.ndarray,
    score_condition: np.ndarray,
    condition_label: str,
    save_path: str,
) -> str:
    from sklearn.metrics import roc_curve, auc as sk_auc

    plt, fig, ax = _new_fig()
    y_true = np.asarray(y_true)

    for label, score, color in (
        ("Original", score_original, "#264653"),
        (condition_label, score_condition, "#e76f51"),
    ):
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUC={sk_auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Original vs {condition_label}")
    ax.legend(loc="lower right")
    return _savefig(plt, fig, save_path)


def plot_original_vs_skin_only(y_true, scores_by_condition: dict, save_path: str) -> str:
    return plot_original_vs_condition_roc(
        y_true, scores_by_condition["original"], scores_by_condition["skin_only"], "Skin-only", save_path
    )


def plot_original_vs_lighting_only(y_true, scores_by_condition: dict, save_path: str) -> str:
    return plot_original_vs_condition_roc(
        y_true, scores_by_condition["original"], scores_by_condition["illum_only"], "Lighting-only", save_path
    )


def plot_original_vs_combined(y_true, scores_by_condition: dict, save_path: str) -> str:
    return plot_original_vs_condition_roc(
        y_true, scores_by_condition["original"], scores_by_condition["both"], "Combined", save_path
    )


# =============================================================================
# 8. Performance-drop comparison
# =============================================================================
def plot_performance_drop_comparison(
    disentanglement_report: dict,
    save_path: str,
    title: str = "Performance drop by counterfactual condition",
) -> str:
    """disentanglement_report: output of
    src.reports.disentanglement_calculator.run_disentanglement_calculator(...)
    Plots Performance_drop_skin / _lighting / _combined side by side with
    the matching ΔAUC as a second series, since a reader needs both to see
    whether accuracy and AUC move the same way."""
    plt, fig, ax = _new_fig(figsize=(7, 5))

    axes_names = ["skin", "lighting", "combined"]
    perf_drop = [disentanglement_report[f"performance_drop_{a}"] for a in axes_names]
    delta_auc = [disentanglement_report[f"delta_auc_{a}"] for a in axes_names]

    x = np.arange(len(axes_names))
    width = 0.35
    ax.bar(x - width / 2, perf_drop, width, label="Performance drop (accuracy)", color="#e9c46a")
    ax.bar(x + width / 2, delta_auc, width, label="ΔAUC", color="#457b9d")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in axes_names])
    ax.set_ylabel("Change vs. original")
    ax.set_title(title)
    ax.legend()
    return _savefig(plt, fig, save_path)


# =============================================================================
# Headline counterfactual-effects figure (mean |Delta_skin| vs |Delta_illum|
# vs interaction residual, with 95% bootstrap CI error bars) -- the one
# figure a reader wants to see the "why" result at a glance.
# =============================================================================
def plot_counterfactual_effects(
    effect_by_identity: dict,
    save_path: str,
    title: str = "Counterfactual sensitivity: Delta_skin vs Delta_illum",
) -> str:
    """effect_by_identity: output of
    src.disentangle.counterfactual_eval.aggregate_by_identity(...)
    (keys: delta_skin, delta_illum, interaction_residual)."""
    from src.utils.metrics import bootstrap_ci

    plt, fig, ax = _new_fig(figsize=(6, 5))

    labels = ["Skin tone", "Illumination", "Interaction"]
    keys = ["delta_skin", "delta_illum", "interaction_residual"]
    colors = ["#e76f51", "#264653", "#2a9d8f"]

    means, los, his = [], [], []
    for k in keys:
        vals = np.abs(np.asarray(effect_by_identity[k]))
        mean, lo, hi = bootstrap_ci(vals)
        means.append(mean)
        los.append(mean - lo)
        his.append(hi - mean)

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[los, his], capsize=5, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean |Delta| (95% bootstrap CI)")
    ax.set_title(title)
    return _savefig(plt, fig, save_path)


# =============================================================================
# Orchestrator: one call, every required figure.
# =============================================================================
def generate_all_required_figures(
    output_dir: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    skin_bins: np.ndarray,
    skin_names: dict[int, str],
    per_skin_table: dict,
    disentanglement_report: dict,
    scores_by_condition: dict[str, np.ndarray] | None = None,
    illum_bins: np.ndarray | None = None,
    illum_names: dict[int, str] | None = None,
    per_illum_table: dict | None = None,
) -> dict[str, str]:
    """Writes every required Week-3/4/5 figure into output_dir and returns
    a dict of {figure_name: path_written}."""
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    paths["roc_by_skin_tone"] = plot_roc_curves_by_group(
        y_true, y_score, skin_bins, skin_names,
        os.path.join(output_dir, "roc_by_skin_tone.png"),
        title="ROC curves by skin-tone group",
    )
    paths["auc_by_skin_tone"] = plot_auc_by_group(
        per_skin_table, os.path.join(output_dir, "auc_by_skin_tone.png"),
        title="AUC by skin-tone group",
    )
    paths["tpr_fpr_by_skin_tone"] = plot_tpr_fpr_by_group(
        per_skin_table, os.path.join(output_dir, "tpr_fpr_by_skin_tone.png"),
        title="TPR / FPR by skin-tone group",
    )
    paths["eer_by_skin_tone"] = plot_eer_by_group(
        per_skin_table, os.path.join(output_dir, "eer_by_skin_tone.png"),
        title="EER by skin-tone group",
    )

    if illum_bins is not None and per_illum_table is not None:
        paths["roc_by_illumination"] = plot_roc_curves_by_group(
            y_true, y_score, illum_bins, illum_names,
            os.path.join(output_dir, "roc_by_illumination.png"),
            title="ROC curves by illumination group",
        )
        paths["auc_by_illumination"] = plot_auc_by_group(
            per_illum_table, os.path.join(output_dir, "auc_by_illumination.png"),
            title="AUC by illumination group",
        )
        paths["tpr_fpr_by_illumination"] = plot_tpr_fpr_by_group(
            per_illum_table, os.path.join(output_dir, "tpr_fpr_by_illumination.png"),
            title="TPR / FPR by illumination group",
        )
        paths["eer_by_illumination"] = plot_eer_by_group(
            per_illum_table, os.path.join(output_dir, "eer_by_illumination.png"),
            title="EER by illumination group",
        )

    if scores_by_condition is not None:
        cf_y_true = disentanglement_report["_per_condition_raw"]["original"].y_true
        paths["original_vs_skin_only"] = plot_original_vs_skin_only(
            cf_y_true, scores_by_condition, os.path.join(output_dir, "original_vs_skin_only.png")
        )
        paths["original_vs_lighting_only"] = plot_original_vs_lighting_only(
            cf_y_true, scores_by_condition, os.path.join(output_dir, "original_vs_lighting_only.png")
        )
        paths["original_vs_combined"] = plot_original_vs_combined(
            cf_y_true, scores_by_condition, os.path.join(output_dir, "original_vs_combined.png")
        )

    paths["performance_drop_comparison"] = plot_performance_drop_comparison(
        disentanglement_report, os.path.join(output_dir, "performance_drop_comparison.png")
    )

    print(f"Wrote {len(paths)} figures to {output_dir}:")
    for name, p in paths.items():
        print(f"  {name}: {p}")
    return paths


if __name__ == "__main__":
    # Smoke test with synthetic data -- writes real PNGs to a temp dir so
    # the plotting code path (not just the numbers) is exercised.
    import tempfile

    from src.utils.metrics import subgroup_metric_table
    from src.reports.disentanglement_calculator import run_disentanglement_calculator

    rng = np.random.default_rng(0)
    n = 600
    y_true = rng.integers(0, 2, size=n)
    skin_bins = rng.integers(0, 6, size=n)
    skin_names = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}
    y_score = np.clip(y_true * 0.6 + rng.normal(0, 0.25, size=n) + 0.2 - 0.03 * skin_bins, 0, 1)

    per_skin_table = subgroup_metric_table(y_true, y_score, skin_bins, skin_names)

    scores_by_condition = {
        "original": y_score,
        "skin_only": np.clip(y_score + rng.normal(0, 0.05, size=n), 0, 1),
        "illum_only": np.clip(y_score + rng.normal(0.08, 0.05, size=n), 0, 1),
        "both": np.clip(y_score + rng.normal(0.1, 0.07, size=n), 0, 1),
    }
    disentanglement_report = run_disentanglement_calculator(y_true, scores_by_condition)

    with tempfile.TemporaryDirectory() as tmp:
        generate_all_required_figures(
            output_dir=tmp,
            y_true=y_true,
            y_score=y_score,
            skin_bins=skin_bins,
            skin_names=skin_names,
            per_skin_table=per_skin_table,
            disentanglement_report=disentanglement_report,
            scores_by_condition=scores_by_condition,
        )
        print("Smoke test OK -- files existed inside tempdir before cleanup.")
