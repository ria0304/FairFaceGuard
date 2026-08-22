"""Week 5 -- finalize disentanglement results.

Pulls together:
    - Week 3 subgroup gap tables (descriptive: THAT a gap exists)
    - Week 4 adversarial accuracy-vs-invariance frontiers
    - Week 4 probe leakage numbers
    - Week 4 counterfactual Delta_skin vs Delta_illum effect sizes (causal:
      WHY the gap exists) with bootstrap CIs and a permutation test
      comparing the two effect sizes directly.

Produces one JSON-serializable report dict that maps directly onto the
paper's Results section (see methodology doc, Section 7 table).
"""

from __future__ import annotations

import json

import numpy as np

from src.utils.metrics import bootstrap_ci, permutation_test_diff


def build_final_report(
    subgroup_eval_results: dict,        # from src/detector/subgroup_eval.py
    sweep_results: dict,                # from src/disentangle/train.py
    leakage_results: dict,              # from src/disentangle/probing.py
    counterfactual_effect_by_identity: dict,  # from counterfactual_eval.aggregate_by_identity
) -> dict:
    delta_skin = counterfactual_effect_by_identity["delta_skin"]
    delta_illum = counterfactual_effect_by_identity["delta_illum"]
    interaction = counterfactual_effect_by_identity["interaction_residual"]

    skin_mean, skin_lo, skin_hi = bootstrap_ci(np.abs(delta_skin))
    illum_mean, illum_lo, illum_hi = bootstrap_ci(np.abs(delta_illum))
    interaction_mean, interaction_lo, interaction_hi = bootstrap_ci(np.abs(interaction))

    p_value = permutation_test_diff(np.abs(delta_skin), np.abs(delta_illum))

    report = {
        "week3_descriptive_gaps": {
            "by_skin_tone": subgroup_eval_results["by_skin_tone"].get("__gap__", {}),
            "by_illumination": subgroup_eval_results["by_illumination"].get("__gap__", {}),
        },
        "week4_accuracy_vs_invariance_frontiers": sweep_results,
        "week4_representation_leakage": leakage_results,
        "week4_causal_counterfactual_effect": {
            "mean_abs_delta_skin": skin_mean,
            "delta_skin_95ci": [skin_lo, skin_hi],
            "mean_abs_delta_illum": illum_mean,
            "delta_illum_95ci": [illum_lo, illum_hi],
            "mean_abs_interaction_residual": interaction_mean,
            "interaction_95ci": [interaction_lo, interaction_hi],
            "permutation_p_delta_skin_vs_delta_illum": p_value,
        },
        "headline_conclusion": _headline(skin_mean, illum_mean, p_value),
    }
    return report


def _headline(skin_mean: float, illum_mean: float, p_value: float, alpha: float = 0.05) -> str:
    if p_value >= alpha:
        return (
            f"No significant difference between |Delta_skin|={skin_mean:.4f} and "
            f"|Delta_illum|={illum_mean:.4f} (p={p_value:.3f}); the data do not "
            f"support attributing the subgroup gap to one factor over the other."
        )
    stronger = "skin tone" if skin_mean > illum_mean else "illumination"
    return (
        f"|Delta_skin|={skin_mean:.4f} vs |Delta_illum|={illum_mean:.4f} "
        f"(p={p_value:.3f}): counterfactual perturbation of {stronger} produces "
        f"the larger shift in detector output, i.e. {stronger} is the stronger "
        f"causal driver of the subgroup gap under this model/dataset."
    )


def save_report(report: dict, path: str) -> None:
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"Not JSON serializable: {type(o)}")

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_default)
    print(f"Final report written to {path}")
