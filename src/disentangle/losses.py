"""Week 4 -- loss functions.

L = L_cls - lambda1 * L_adv(h1) - lambda2 * L_adv(h2)

The minus signs are already implemented by the GRL (it flips the gradient
sign during backward). Do NOT also negate the adversarial losses here --
compute them with a normal positive cross-entropy; the GRL handles the sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DisentangleLossWeights:
    cls: float = 1.0
    skin_adv: float = 1.0
    illum_adv: float = 1.0


def disentangle_loss(
    outputs: dict,
    fake_labels: torch.Tensor,
    skin_labels: torch.Tensor,
    illum_labels: torch.Tensor,
    weights: DisentangleLossWeights = DisentangleLossWeights(),
) -> dict:
    l_cls = F.cross_entropy(outputs["fake_logits"], fake_labels)
    l_skin = F.cross_entropy(outputs["skin_logits"], skin_labels)
    l_illum = F.cross_entropy(outputs["illum_logits"], illum_labels)

    total = weights.cls * l_cls + weights.skin_adv * l_skin + weights.illum_adv * l_illum
    return {"total": total, "cls": l_cls, "skin_adv": l_skin, "illum_adv": l_illum}
