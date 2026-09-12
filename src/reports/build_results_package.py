"""Week 5 -- final results package + reproducibility bundle .

ONE call, `build_final_results_package(...)`, from ONE script, produces the
whole deliverable:

    results/
      baseline_metrics.csv
      subgroup_metrics.csv
      counterfactual_metrics.csv
      statistical_tests.csv
      predictions.csv
      dataset_split.csv
      experiment_config.json
      model_checkpoint.pt            (copied in, if a checkpoint path is given)
      figures/
        subgroup_auc.png
        tpr_fpr_gap.png
        eer_by_group.png
        counterfactual_effects.png

This is deliberately a thin *assembler*: every number in it was already
computed by an earlier stage (week3_metric_report, subgroup_eval,
disentanglement_calculator, statistical_validation, counterfactual_eval).
Nothing is recomputed here, so this script can be re-run cheaply to
repackage results without re-running the model.

Reproducibility (item 7) is treated as "everything a second person would
need to exactly reproduce or audit this run", all captured in
`experiment_config.json` and the sibling files above:
    - random seed(s)
    - full experiment configuration (hyperparameters, paths, versions)
    - dataset split assignment (which face_id went to train/val/test)
    - model checkpoint (copied into the package, not just referenced)
    - raw predictions (for independent re-scoring)
    - metrics as CSV (not just JSON, for spreadsheet/stats-software use)
    - statistical-test results as CSV
    - every generated figure
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np

from src.reports.generate_figures import (
    plot_auc_by_group,
    plot_tpr_fpr_by_group,
    plot_eer_by_group,
    plot_counterfactual_effects,
)


# =============================================================================
# Individual CSV writers -- each is also usable standalone.
# =============================================================================
def write_baseline_metrics_csv(overall_metrics: dict, path: str) -> str:
    """overall_metrics: week3_report["overall"] from
    src.reports.week3_metric_report.build_week3_report(...)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in overall_metrics.items():
            writer.writerow([k, v])
    return path


def write_subgroup_metrics_csv(subgroup_results: dict, path: str) -> str:
    """subgroup_results: output of src.detector.subgroup_eval.run_subgroup_eval(...)
    (by_skin_tone / by_illumination tables, each name -> metric dict, plus a
    "__gap__" row)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fieldnames = ["axis", "group", "n", "accuracy", "auc", "eer", "precision", "recall", "f1", "fpr", "fnr"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for axis_key, axis_label in (("by_skin_tone", "skin_tone"), ("by_illumination", "illumination")):
            table = subgroup_results.get(axis_key, {})
            for group_name, m in table.items():
                row = {"axis": axis_label, "group": group_name}
                row.update({k: m.get(k, "") for k in fieldnames if k not in ("axis", "group")})
                writer.writerow(row)
    return path


def write_counterfactual_metrics_csv(disentanglement_report: dict, path: str) -> str:
    """disentanglement_report: output of
    src.reports.disentanglement_calculator.run_disentanglement_calculator(...)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "n", "auc", "accuracy", "tpr", "fpr", "eer"])
        for cond, m in disentanglement_report["per_condition_metrics"].items():
            writer.writerow([cond, m["n"], m["auc"], m["accuracy"], m["tpr"], m["fpr"], m["eer"]])

        writer.writerow([])
        writer.writerow(["axis", "delta_auc", "performance_drop", "tpr_change", "fpr_change", "eer_change"])
        for axis in ("skin", "lighting", "combined"):
            d = disentanglement_report["tpr_fpr_eer_changes"][axis]
            writer.writerow([
                axis,
                disentanglement_report[f"delta_auc_{axis}"],
                disentanglement_report[f"performance_drop_{axis}"],
                d[f"tpr_change_{axis}"],
                d[f"fpr_change_{axis}"],
                d[f"eer_change_{axis}"],
            ])
    return path


def write_statistical_tests_csv(stats_report: dict, path: str) -> str:
    """stats_report: output of
    src.reports.statistical_validation.run_pre_specified_test_plan(...)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fieldnames = [
        "comparison", "description", "skipped", "point_estimate", "ci_low", "ci_high",
        "p_value", "corrected_p_value", "effect_size", "significant_after_correction",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, r in stats_report["comparisons"].items():
            if r["skipped"]:
                writer.writerow({"comparison": name, "description": r["description"], "skipped": True})
                continue
            ci = r.get("ci_95") or r.get("ci_95_of_difference") or [None, None]
            writer.writerow({
                "comparison": name,
                "description": r["description"],
                "skipped": False,
                "point_estimate": r["point_estimate"],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "p_value": r.get("p_value"),
                "corrected_p_value": r.get("corrected_p_value"),
                "effect_size": r.get("effect_size", r.get("effect_size_cohen_d")),
                "significant_after_correction": r["significant_after_correction"],
            })
    return path


def write_predictions_csv(subgroup_results: dict, path: str) -> str:
    """subgroup_results: output of run_subgroup_eval(...), using the raw
    "_y_true"/"_y_score"/"_skin_bins"/"_illum_bins"/"_face_ids" arrays it
    carries alongside the summary tables."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    y_true = subgroup_results["_y_true"]
    y_score = subgroup_results["_y_score"]
    skin_bins = subgroup_results["_skin_bins"]
    illum_bins = subgroup_results["_illum_bins"]
    face_ids = subgroup_results.get("_face_ids") or [f"sample_{i}" for i in range(len(y_true))]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["face_id", "y_true", "y_score", "y_pred", "skin_bin", "illum_bin"])
        for fid, yt, ys, sb, ib in zip(face_ids, y_true, y_score, skin_bins, illum_bins):
            writer.writerow([fid, int(yt), float(ys), int(ys >= 0.5), int(sb), int(ib)])
    return path


def write_dataset_split_csv(dataset_split, path: str) -> str:
    """dataset_split: a pandas DataFrame with at least face_id/split columns
    (e.g. the output of src.data.identity_split), OR a path to an existing
    labels_with_splits.csv to copy in directly."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if isinstance(dataset_split, str):
        shutil.copy(dataset_split, path)
    else:
        dataset_split.to_csv(path, index=False)
    return path


def write_experiment_config_json(
    seed: int,
    config: dict,
    path: str,
    extra_seeds: list[int] | None = None,
    checkpoint_path: str | None = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "multi_seed_replication_seeds": extra_seeds or [],
        "checkpoint_path": checkpoint_path,
        "config": config,
    }

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_default)
    return path


# =============================================================================
# The single entry point.
# =============================================================================
def build_final_results_package(
    output_dir: str,
    seed: int,
    config: dict,
    week3_report: dict,
    subgroup_results: dict,
    disentanglement_report: dict,
    stats_report: dict,
    effect_by_identity: dict,
    checkpoint_path: str | None = None,
    dataset_split=None,
    extra_seeds: list[int] | None = None,
) -> dict[str, str]:
    """Writes the entire results/ package described in the module
    docstring and returns a dict of {artifact_name: path_written}.

    Every argument is the direct output of an already-implemented pipeline
    stage:
        week3_report            <- src.reports.week3_metric_report.build_week3_report
        subgroup_results        <- src.detector.subgroup_eval.run_subgroup_eval
        disentanglement_report  <- src.reports.disentanglement_calculator.run_disentanglement_calculator
        stats_report            <- src.reports.statistical_validation.run_pre_specified_test_plan
        effect_by_identity      <- src.disentangle.counterfactual_eval.aggregate_by_identity
        dataset_split           <- a DataFrame (or CSV path) with a 'split' column,
                                    e.g. src.data.identity_split output
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    paths: dict[str, str] = {}
    paths["baseline_metrics"] = write_baseline_metrics_csv(
        week3_report["overall"], os.path.join(output_dir, "baseline_metrics.csv")
    )
    paths["subgroup_metrics"] = write_subgroup_metrics_csv(
        subgroup_results, os.path.join(output_dir, "subgroup_metrics.csv")
    )
    paths["counterfactual_metrics"] = write_counterfactual_metrics_csv(
        disentanglement_report, os.path.join(output_dir, "counterfactual_metrics.csv")
    )
    paths["statistical_tests"] = write_statistical_tests_csv(
        stats_report, os.path.join(output_dir, "statistical_tests.csv")
    )
    paths["predictions"] = write_predictions_csv(
        subgroup_results, os.path.join(output_dir, "predictions.csv")
    )

    if dataset_split is not None:
        paths["dataset_split"] = write_dataset_split_csv(
            dataset_split, os.path.join(output_dir, "dataset_split.csv")
        )

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        dest = os.path.join(output_dir, "model_checkpoint.pt")
        shutil.copy(checkpoint_path, dest)
        paths["model_checkpoint"] = dest

    paths["experiment_config"] = write_experiment_config_json(
        seed=seed,
        config=config,
        path=os.path.join(output_dir, "experiment_config.json"),
        extra_seeds=extra_seeds,
        checkpoint_path=paths.get("model_checkpoint"),
    )

    paths["fig_subgroup_auc"] = plot_auc_by_group(
        subgroup_results["by_skin_tone"], os.path.join(figures_dir, "subgroup_auc.png"),
        title="AUC by skin-tone group",
    )
    paths["fig_tpr_fpr_gap"] = plot_tpr_fpr_by_group(
        subgroup_results["by_skin_tone"], os.path.join(figures_dir, "tpr_fpr_gap.png"),
        title="TPR / FPR by skin-tone group",
    )
    paths["fig_eer_by_group"] = plot_eer_by_group(
        subgroup_results["by_skin_tone"], os.path.join(figures_dir, "eer_by_group.png"),
        title="EER by skin-tone group",
    )
    paths["fig_counterfactual_effects"] = plot_counterfactual_effects(
        effect_by_identity, os.path.join(figures_dir, "counterfactual_effects.png")
    )

    print(f"Final results package written to {output_dir}/:")
    for name, p in paths.items():
        print(f"  {name}: {p}")
    return paths


if __name__ == "__main__":
    # Smoke test with synthetic data -- exercises every writer end to end.
    import tempfile

    import pandas as pd

    from src.utils.metrics import subgroup_metric_table
    from src.reports.disentanglement_calculator import run_disentanglement_calculator
    from src.reports.statistical_validation import run_pre_specified_test_plan
    from src.utils.seed import set_seed

    set_seed(42)
    rng = np.random.default_rng(42)
    n = 400

    y_true = rng.integers(0, 2, size=n)
    skin_bins = rng.integers(0, 6, size=n)
    illum_bins = rng.integers(0, 5, size=n)
    y_score = np.clip(y_true * 0.6 + rng.normal(0, 0.25, size=n) + 0.2 - 0.02 * skin_bins, 0, 1)
    face_ids = [f"face_{i:04d}" for i in range(n)]

    skin_names = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}
    illum_names = {0: "very_dark", 1: "dark", 2: "mid", 3: "bright", 4: "very_bright"}
    subgroup_results = {
        "by_skin_tone": subgroup_metric_table(y_true, y_score, skin_bins, skin_names),
        "by_illumination": subgroup_metric_table(y_true, y_score, illum_bins, illum_names),
        "_y_true": y_true, "_y_score": y_score,
        "_skin_bins": skin_bins, "_illum_bins": illum_bins, "_face_ids": face_ids,
    }

    week3_report = {"overall": {
        "n": n, "roc_auc": 0.9, "accuracy": 0.85, "precision": 0.83,
        "recall_tpr": 0.86, "f1": 0.84, "fpr": 0.12, "fnr": 0.14, "eer": 0.13,
    }}

    scores_by_condition = {
        "original": y_score,
        "skin_only": np.clip(y_score + rng.normal(0, 0.05, size=n), 0, 1),
        "illum_only": np.clip(y_score + rng.normal(0.08, 0.05, size=n), 0, 1),
        "both": np.clip(y_score + rng.normal(0.1, 0.07, size=n), 0, 1),
    }
    disentanglement_report = run_disentanglement_calculator(y_true, scores_by_condition)

    effect_by_identity = {
        "delta_skin": rng.normal(0.02, 0.05, size=n),
        "delta_illum": rng.normal(0.08, 0.05, size=n),
        "interaction_residual": rng.normal(0.0, 0.02, size=n),
    }
    stats_report = run_pre_specified_test_plan(
        effect_by_identity=effect_by_identity,
        per_condition_raw=disentanglement_report["_per_condition_raw"],
    )

    dataset_split = pd.DataFrame({
        "face_id": face_ids,
        "split": rng.choice(["train", "val", "test"], size=n, p=[0.7, 0.15, 0.15]),
    })

    with tempfile.TemporaryDirectory() as tmp:
        build_final_results_package(
            output_dir=os.path.join(tmp, "results"),
            seed=42,
            config={"backbone": "efficientnet_b4", "epochs": 20, "batch_size": 32},
            week3_report=week3_report,
            subgroup_results=subgroup_results,
            disentanglement_report=disentanglement_report,
            stats_report=stats_report,
            effect_by_identity=effect_by_identity,
            dataset_split=dataset_split,
            extra_seeds=[123, 2024, 3407, 7777],
        )
        print("\nSmoke test OK.")
