"""Week 5 -- statistical validation, run against a PRE-SPECIFIED test plan.

The methodological requirement this module exists to satisfy: the test to
run, the comparisons to make, the sidedness, and the alpha/correction
method must all be fixed *before* looking at results. Choosing a test
after seeing which one gives a smaller p-value ("p-hacking by test
selection") invalidates the p-value. So PRE_SPECIFIED_TEST_PLAN below is
the whole analysis plan, written down once, and every function in this
module just executes it -- nothing here branches on the observed data to
decide which test to use.

For each pre-specified comparison this reports, automatically:
    - p-value
    - confidence interval
    - effect size
    - significance decision (after multiple-comparison correction)

Comparisons draw on two upstream artifacts, both already computed
elsewhere in the pipeline:
    - `effect_by_identity` from src/disentangle/counterfactual_eval.py
      (per-identity delta_skin / delta_illum / interaction_residual)
    - the per-condition raw arrays from
      src/reports/disentanglement_calculator.py's
      run_disentanglement_calculator(...)["_per_condition_raw"]
      (per-sample correctness at threshold 0.5, for the performance-drop
      comparisons)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.metrics import (
    bootstrap_ci,
    paired_permutation_test,
    effect_size_cohen_d,
    bonferroni_correction,
    benjamini_hochberg_correction,
)


@dataclass(frozen=True)
class PreSpecifiedComparison:
    """One entry in the fixed analysis plan. `kind` selects which of the
    fixed procedures below runs for it -- never chosen based on results."""

    name: str
    kind: str  # "bootstrap_ci_abs_nonzero" | "paired_permutation" | "paired_permutation_correct"
    description: str
    one_sided: bool = False


# =============================================================================
# THE PRE-SPECIFIED TEST PLAN
# Registered before any results are inspected. Do not add, remove, or
# reorder entries based on what the data turns out to show; that defeats
# the purpose of pre-specification. Extend it only for a new research
# question decided in advance of running the experiment.
# =============================================================================
PRE_SPECIFIED_TEST_PLAN: tuple[PreSpecifiedComparison, ...] = (
    PreSpecifiedComparison(
        name="delta_skin_nonzero",
        kind="bootstrap_ci_abs_nonzero",
        description="Is the counterfactual skin-tone effect (|Delta_skin|) different from zero?",
    ),
    PreSpecifiedComparison(
        name="delta_illum_nonzero",
        kind="bootstrap_ci_abs_nonzero",
        description="Is the counterfactual illumination effect (|Delta_illum|) different from zero?",
    ),
    PreSpecifiedComparison(
        name="delta_skin_vs_delta_illum",
        kind="paired_permutation",
        description="Does |Delta_skin| differ from |Delta_illum| on the same identities? "
        "(the headline causal comparison: which factor drives the gap more)",
    ),
    PreSpecifiedComparison(
        name="performance_drop_skin_vs_lighting",
        kind="paired_permutation_correct",
        description="Does per-sample correctness under skin_only differ from correctness "
        "under illum_only, on the same identities?",
    ),
    PreSpecifiedComparison(
        name="performance_drop_skin_vs_combined",
        kind="paired_permutation_correct",
        description="Does per-sample correctness under skin_only differ from correctness "
        "under both (combined), on the same identities?",
    ),
)

ALPHA = 0.05
CORRECTION_METHOD = "bonferroni"  # fixed in advance; alternative "benjamini_hochberg"
N_BOOT = 2000
N_PERM = 5000
SEED = 0


def _run_bootstrap_ci_abs_nonzero(values: np.ndarray) -> dict:
    mean, lo, hi = bootstrap_ci(np.abs(values), n_boot=N_BOOT, seed=SEED)
    # A CI for |Delta| excluding 0 is the standard evidence-of-nonzero-effect
    # decision rule for this one-sided-by-construction quantity.
    p_like_significant = lo > 0
    return {
        "point_estimate": mean,
        "ci_95": [lo, hi],
        "p_value": None,  # bootstrap CI reported instead of a p-value by design
        "effect_size": mean,  # mean |Delta| is itself the effect-size unit here
        "decision_rule": "95% CI excludes 0",
        "raw_significant": bool(p_like_significant),
    }


def _run_paired_permutation(a: np.ndarray, b: np.ndarray) -> dict:
    p = paired_permutation_test(a, b, n_perm=N_PERM, seed=SEED)
    d = effect_size_cohen_d(a, b)
    mean, lo, hi = bootstrap_ci(a - b, n_boot=N_BOOT, seed=SEED)
    return {
        "point_estimate": float(np.mean(a) - np.mean(b)),
        "ci_95_of_difference": [lo, hi],
        "p_value": p,
        "effect_size_cohen_d": d,
        "decision_rule": f"paired permutation p < {ALPHA} (before correction)",
        "raw_significant": bool(p < ALPHA),
    }


def run_pre_specified_test_plan(
    effect_by_identity: dict | None = None,
    per_condition_raw: dict | None = None,
) -> dict:
    """Executes PRE_SPECIFIED_TEST_PLAN exactly as registered above.

    effect_by_identity: output of
        src.disentangle.counterfactual_eval.aggregate_by_identity(...)
        (keys: delta_skin, delta_illum, interaction_residual)
    per_condition_raw: the "_per_condition_raw" entry of
        src.reports.disentanglement_calculator.run_disentanglement_calculator(...)
        (keys: original/skin_only/illum_only/both -> ConditionMetrics with
        a `.correct` per-sample 0/1 array)

    Either argument may be omitted if the corresponding comparisons aren't
    applicable yet; those entries are skipped (not silently marked
    significant/non-significant) and flagged as "skipped" in the output.
    """
    results: dict[str, dict] = {}

    for comp in PRE_SPECIFIED_TEST_PLAN:
        try:
            if comp.kind == "bootstrap_ci_abs_nonzero":
                if effect_by_identity is None:
                    raise KeyError("effect_by_identity not provided")
                key = "delta_skin" if comp.name == "delta_skin_nonzero" else "delta_illum"
                out = _run_bootstrap_ci_abs_nonzero(np.asarray(effect_by_identity[key]))

            elif comp.kind == "paired_permutation":
                if effect_by_identity is None:
                    raise KeyError("effect_by_identity not provided")
                a = np.abs(np.asarray(effect_by_identity["delta_skin"]))
                b = np.abs(np.asarray(effect_by_identity["delta_illum"]))
                out = _run_paired_permutation(a, b)

            elif comp.kind == "paired_permutation_correct":
                if per_condition_raw is None:
                    raise KeyError("per_condition_raw not provided")
                cond_a, cond_b = (
                    ("skin_only", "illum_only")
                    if comp.name == "performance_drop_skin_vs_lighting"
                    else ("skin_only", "both")
                )
                a = per_condition_raw[cond_a].correct.astype(float)
                b = per_condition_raw[cond_b].correct.astype(float)
                min_len = min(len(a), len(b))
                out = _run_paired_permutation(a[:min_len], b[:min_len])

            else:
                raise ValueError(f"Unknown comparison kind: {comp.kind}")

            out["skipped"] = False
        except KeyError as e:
            out = {"skipped": True, "reason": str(e)}

        out["name"] = comp.name
        out["description"] = comp.description
        results[comp.name] = out

    # Multiple-comparison correction across every comparison that actually
    # ran and has a p-value (the bootstrap-CI entries don't produce one by
    # design, so they're excluded from the correction and evaluated purely
    # on their CI).
    tested_names = [
        name for name, r in results.items() if not r["skipped"] and r.get("p_value") is not None
    ]
    raw_p_values = [results[name]["p_value"] for name in tested_names]

    if raw_p_values:
        if CORRECTION_METHOD == "bonferroni":
            corrected, sig = bonferroni_correction(raw_p_values, alpha=ALPHA)
        else:
            corrected, sig = benjamini_hochberg_correction(raw_p_values, alpha=ALPHA)
        for name, p_corr, is_sig in zip(tested_names, corrected, sig):
            results[name]["corrected_p_value"] = p_corr
            results[name]["significant_after_correction"] = bool(is_sig)

    # CI-based entries get their significance decision directly (no
    # p-value to correct), but we still surface a final decision field for
    # a uniform report shape.
    for name, r in results.items():
        if r["skipped"]:
            r["significant_after_correction"] = None
            continue
        if "significant_after_correction" not in r:
            r["significant_after_correction"] = r["raw_significant"]

    return {
        "alpha": ALPHA,
        "multiple_comparison_correction": CORRECTION_METHOD,
        "n_bootstrap": N_BOOT,
        "n_permutations": N_PERM,
        "comparisons": results,
    }


def print_statistical_report(report: dict) -> None:
    print("=== Statistical validation (pre-specified test plan) ===")
    print(f"alpha={report['alpha']}  correction={report['multiple_comparison_correction']}")
    for name, r in report["comparisons"].items():
        print(f"\n-- {name} --")
        print(f"  {r['description']}")
        if r["skipped"]:
            print(f"  SKIPPED: {r['reason']}")
            continue
        print(f"  point estimate       : {r['point_estimate']:.4f}")
        if "ci_95" in r:
            print(f"  95% CI               : [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]")
        if "ci_95_of_difference" in r:
            print(f"  95% CI of difference : [{r['ci_95_of_difference'][0]:.4f}, {r['ci_95_of_difference'][1]:.4f}]")
        if r["p_value"] is not None:
            print(f"  p-value              : {r['p_value']:.4f}")
            print(f"  corrected p-value    : {r.get('corrected_p_value', float('nan')):.4f}")
        if "effect_size" in r:
            print(f"  effect size          : {r['effect_size']:.4f}")
        if "effect_size_cohen_d" in r:
            print(f"  effect size (d)      : {r['effect_size_cohen_d']:.4f}")
        decision = "SIGNIFICANT" if r["significant_after_correction"] else "not significant"
        print(f"  decision             : {decision}")


def save_statistical_report(report: dict, path: str) -> None:
    import json

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_default)
    print(f"Statistical validation report written to {path}")


if __name__ == "__main__":
    # Smoke test with synthetic data.
    rng = np.random.default_rng(0)
    n = 300
    effect_by_identity = {
        "delta_skin": rng.normal(0.02, 0.05, size=n),
        "delta_illum": rng.normal(0.08, 0.05, size=n),
        "interaction_residual": rng.normal(0.0, 0.02, size=n),
    }

    from src.reports.disentanglement_calculator import run_disentanglement_calculator

    y_true = rng.integers(0, 2, size=n)
    base = np.clip(y_true * 0.6 + rng.normal(0, 0.25, size=n) + 0.2, 0, 1)
    scores = {
        "original": base,
        "skin_only": np.clip(base + rng.normal(0, 0.05, size=n), 0, 1),
        "illum_only": np.clip(base + rng.normal(0.08, 0.05, size=n), 0, 1),
        "both": np.clip(base + rng.normal(0.1, 0.07, size=n), 0, 1),
    }
    calc_report = run_disentanglement_calculator(y_true, scores)

    report = run_pre_specified_test_plan(
        effect_by_identity=effect_by_identity,
        per_condition_raw=calc_report["_per_condition_raw"],
    )
    print_statistical_report(report)
