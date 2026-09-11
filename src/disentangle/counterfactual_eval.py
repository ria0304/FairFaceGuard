"""Week 4 -- counterfactual invariance testing (methodology Section 5.3).

THE HEADLINE RESULT. Runs on the FROZEN Week-3 baseline detector using the
factorial {original, skin_only, illum_only, both} sets produced by Week 2's
augmentation pipeline. Because identity, texture, and generation artifacts
are held fixed and only the targeted variable is perturbed, Delta_skin vs.
Delta_illum is a direct causal comparison -- this is what actually answers
the core research question, on the model as deployed (not a retrained one).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CounterfactualBatch:
    original: torch.Tensor
    skin_only: torch.Tensor
    illum_only: torch.Tensor
    both: torch.Tensor
    face_ids: list[str]


@torch.no_grad()
def counterfactual_effect(frozen_model, cf: CounterfactualBatch, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    """Returns per-sample Delta_skin / Delta_illum / interaction_residual
    arrays plus aggregate summary stats.

    interaction_residual = p(both) - [p(orig) + Delta_skin + Delta_illum]
    A large residual flags an interaction effect (H_artifact-interaction),
    i.e. detection difficulty that depends on the SKIN-TONE x ILLUMINATION
    combination rather than either alone.
    """
    frozen_model.eval().to(device)

    def p_fake(x: torch.Tensor) -> np.ndarray:
        x = x.to(device)
        out = frozen_model(x)
        logits = out["fake_logits"] if isinstance(out, dict) else out
        return F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    p_orig = p_fake(cf.original)
    p_skin = p_fake(cf.skin_only)
    p_illum = p_fake(cf.illum_only)
    p_both = p_fake(cf.both)

    delta_skin = p_skin - p_orig
    delta_illum = p_illum - p_orig
    additive_pred = p_orig + delta_skin + delta_illum
    interaction_residual = p_both - additive_pred

    return {
        "face_ids": cf.face_ids,
        "delta_skin": delta_skin,
        "delta_illum": delta_illum,
        "interaction_residual": interaction_residual,
        "mean_abs_delta_skin": float(np.mean(np.abs(delta_skin))),
        "mean_abs_delta_illum": float(np.mean(np.abs(delta_illum))),
        "mean_abs_interaction": float(np.mean(np.abs(interaction_residual))),
    }


def aggregate_by_identity(effect: dict) -> dict:
    """Collapse per-sample effects to one value per identity (mean), which
    is what should be fed to bootstrap_ci / permutation_test_diff in
    src/utils/metrics.py to avoid inflated confidence from correlated frames
    of the same face."""
    ids = np.array(effect["face_ids"])
    unique_ids = np.unique(ids)
    out = {"face_id": [], "delta_skin": [], "delta_illum": [], "interaction_residual": []}
    for uid in unique_ids:
        mask = ids == uid
        out["face_id"].append(uid)
        out["delta_skin"].append(effect["delta_skin"][mask].mean())
        out["delta_illum"].append(effect["delta_illum"][mask].mean())
        out["interaction_residual"].append(effect["interaction_residual"][mask].mean())
    return {k: (np.array(v) if k != "face_id" else v) for k, v in out.items()}
