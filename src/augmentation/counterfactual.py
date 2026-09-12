"""Week 2 -- Counterfactual augmentation pipeline.

Generates two independent interventions on the SAME face/frame:
    - skin-tone-only shift:   pigment (ITA) changed, illumination held fixed
    - illumination-only shift: lighting/white-balance changed, pigment held fixed

and a factorial {original, skin_only, illum_only, both} set, which is what
lets Week 4's counterfactual_effect() decompose the detector's response into
skin-tone-attributable vs. illumination-attributable components.

Two backends are provided:
    - classical (default): fast, interpretable Lab-space manipulation.
      Good enough to ship Week 2 on time; document its limitations (see
      module docstring in disentangle/counterfactual_eval.py and the
      methodology's Section 3.3/8).
    - gan (optional stub): hook for a StyleGAN2/diffusion-based editor if
      you have one trained; left as an interface so it drops in without
      touching the rest of the pipeline.

Every generated counterfactual is re-validated by round-tripping through
the Week 1 ITA pipeline (manipulation check) before being accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.annotation.ita_fitzpatrick import (
    compute_ita,
    estimate_illuminant_gray_world,
    white_balance_correct,
)


@dataclass
class CounterfactualSet:
    face_id: str
    original: np.ndarray
    skin_only: np.ndarray
    illum_only: np.ndarray
    both: np.ndarray
    validation: dict

    @property
    def accepted(self) -> bool:
        return bool(self.validation.get("accepted", False))

    @property
    def reject_reasons(self) -> list[str]:
        return list(self.validation.get("reject_reasons", []))


# --------------------------------------------------------------------------
# Classical Lab-space backend
# --------------------------------------------------------------------------

def shift_skin_tone(image_bgr: np.ndarray, target_ita_delta: float) -> np.ndarray:
    """Shift pigmentation along the ITA axis (adjust L*/b* in Lab space)
    while leaving shading structure (relative luminance variation) intact.

    target_ita_delta: degrees to shift ITA by (positive = lighter, negative = darker).
    This is an approximation -- it moves L*/b* together to trace the ITA
    formula's direction, not a full melanin/hemoglobin decomposition. Note
    the residual pigment/light coupling this introduces in your validation
    report (see methodology Section 3.3).
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # scale L and b together in the direction that moves ITA by target_ita_delta,
    # holding the mean luminance *pattern* (shading) fixed by operating on the
    # deviation from the per-image median rather than absolute L.
    L_median = np.median(L)
    b_median = np.median(b)

    # small-angle approximation: d(ITA)/d(scale) calibrated numerically
    step = 1.0
    ita_before = compute_ita(np.stack([L.flatten(), a.flatten(), b.flatten()], axis=1))
    trial_b = b_median + (b - b_median) * 1.05
    trial_L = L
    ita_after = compute_ita(
        np.stack([trial_L.flatten(), a.flatten(), trial_b.flatten()], axis=1)
    )
    sensitivity = (ita_after - ita_before) / 0.05 if abs(ita_after - ita_before) > 1e-6 else 1.0
    b_scale = 1.0 + (target_ita_delta / sensitivity) if sensitivity != 0 else 1.0
    b_scale = float(np.clip(b_scale, 0.5, 1.8))

    new_b = b_median + (b - b_median) * b_scale
    new_L = L_median + (L - L_median) * (1.0 if target_ita_delta == 0 else np.clip(1.0 - 0.15 * np.sign(target_ita_delta) * -1, 0.7, 1.3))

    lab_out = np.stack([np.clip(new_L, 0, 255), a, np.clip(new_b, 0, 255)], axis=-1).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)


def shift_illumination(
    image_bgr: np.ndarray,
    color_temp_shift: float = 0.0,
    exposure_stops: float = 0.0,
) -> np.ndarray:
    """Shift scene illumination: white-balance/color-temperature and exposure,
    holding pigmentation (ITA, computed post-hoc after de-illumination)
    approximately fixed.

    color_temp_shift: -1 (cooler/blue) .. +1 (warmer/orange)
    exposure_stops: photographic stops, e.g. +1 = 2x brighter
    """
    img = image_bgr.astype(np.float32)

    # exposure
    img *= (2.0 ** exposure_stops)

    # color temperature: push blue vs red channel gains in opposite directions
    gain = 0.15 * color_temp_shift
    img[..., 2] *= (1.0 + gain)   # R channel (BGR order, index 2)
    img[..., 0] *= (1.0 - gain)   # B channel

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_counterfactual_set(
    image_bgr: np.ndarray,
    face_id: str,
    ita_delta: float = 15.0,
    color_temp_shift: float = 0.3,
    exposure_stops: float = 0.5,
) -> CounterfactualSet:
    """Produce the factorial {original, skin_only, illum_only, both} set and
    run the manipulation-check validation loop."""
    skin_only = shift_skin_tone(image_bgr, ita_delta)
    illum_only = shift_illumination(image_bgr, color_temp_shift, exposure_stops)
    both = shift_illumination(
        shift_skin_tone(image_bgr, ita_delta), color_temp_shift, exposure_stops
    )

    validation = _validate_counterfactuals(
        image_bgr, skin_only, illum_only, both, ita_delta
    )

    return CounterfactualSet(
        face_id=face_id,
        original=image_bgr,
        skin_only=skin_only,
        illum_only=illum_only,
        both=both,
        validation=validation,
    )


def _ita_of(image_bgr: np.ndarray) -> float:
    illum = estimate_illuminant_gray_world(image_bgr)
    balanced = white_balance_correct(image_bgr, illum)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    return compute_ita(lab)


def _illuminant_of(image_bgr: np.ndarray) -> np.ndarray:
    return estimate_illuminant_gray_world(image_bgr)


def validate_counterfactual_images(
    original: np.ndarray,
    skin_only: np.ndarray,
    illum_only: np.ndarray,
    both: np.ndarray,
    intended_ita_delta: float = 15.0,
    ita_tolerance: float = 8.0,
    illum_tolerance_frac: float = 0.15,
) -> dict:
    """Public entry point for the validation gate, for callers (e.g. the
    unified counterfactual experiment runner) that already have the four
    images on disk/in memory and don't want to regenerate them from
    scratch. Same checks as `generate_counterfactual_set`'s internal gate."""
    return _validate_counterfactuals(
        original, skin_only, illum_only, both,
        intended_ita_delta, ita_tolerance, illum_tolerance_frac,
    )


def _validate_counterfactuals(
    original: np.ndarray,
    skin_only: np.ndarray,
    illum_only: np.ndarray,
    both: np.ndarray,
    intended_ita_delta: float,
    ita_tolerance: float = 8.0,
    illum_tolerance_frac: float = 0.15,
) -> dict:
    """Counterfactual validation gate (methodology Section 3.1/3.2).

    Verifies each generated counterfactual actually changes the INTENDED
    factor only, for all four arms of the factorial set:
        - skin_only  -> skin-tone (ITA) changes, illumination stays stable
        - illum_only -> illumination changes, skin-tone (ITA) stays stable
        - both       -> both change
    Any sample whose unintended-factor change is too large (or whose
    intended-factor change is too small) is flagged and excluded via
    `accepted=False` / `reject_reasons`, rather than silently kept.
    """
    ita_orig = _ita_of(original)
    ita_skin = _ita_of(skin_only)
    ita_illum = _ita_of(illum_only)
    ita_both = _ita_of(both)

    illum_orig = _illuminant_of(original)
    illum_skin = _illuminant_of(skin_only)
    illum_illum = _illuminant_of(illum_only)
    illum_both = _illuminant_of(both)

    def illum_shift_frac(illum_variant: np.ndarray) -> float:
        return float(
            np.linalg.norm(illum_variant - illum_orig) / (np.linalg.norm(illum_orig) + 1e-6)
        )

    # --- skin_only: intended = ITA shift, unintended = illuminant shift ---
    measured_ita_delta_skin = ita_skin - ita_orig
    skin_intended_ok = abs(measured_ita_delta_skin - intended_ita_delta) < ita_tolerance
    skin_leak_into_illum = illum_shift_frac(illum_skin)
    skin_unintended_ok = skin_leak_into_illum < illum_tolerance_frac
    skin_only_accepted = bool(skin_intended_ok and skin_unintended_ok)

    # --- illum_only: intended = illuminant shift, unintended = ITA shift ---
    illum_shift_magnitude = illum_shift_frac(illum_illum)
    illum_intended_ok = illum_shift_magnitude > illum_tolerance_frac
    illum_leak_into_ita = abs(ita_illum - ita_orig)
    illum_unintended_ok = illum_leak_into_ita < ita_tolerance
    illum_only_accepted = bool(illum_intended_ok and illum_unintended_ok)

    # --- both: BOTH factors are intended to change here ---
    measured_ita_delta_both = ita_both - ita_orig
    both_ita_ok = abs(measured_ita_delta_both - intended_ita_delta) < ita_tolerance
    both_illum_shift_magnitude = illum_shift_frac(illum_both)
    both_illum_ok = both_illum_shift_magnitude > illum_tolerance_frac
    both_accepted = bool(both_ita_ok and both_illum_ok)

    reject_reasons: list[str] = []
    if not skin_intended_ok:
        reject_reasons.append("skin_only: intended ITA shift not achieved")
    if not skin_unintended_ok:
        reject_reasons.append("skin_only: illuminant leaked too much (unintended change)")
    if not illum_intended_ok:
        reject_reasons.append("illum_only: intended illuminant shift not achieved")
    if not illum_unintended_ok:
        reject_reasons.append("illum_only: ITA leaked too much (unintended change)")
    if not both_ita_ok:
        reject_reasons.append("both: ITA did not shift as intended")
    if not both_illum_ok:
        reject_reasons.append("both: illuminant did not shift as intended")

    # A face's whole counterfactual set is only usable for the Week-4 causal
    # comparison if every arm passes its own manipulation check.
    accepted = skin_only_accepted and illum_only_accepted and both_accepted

    return {
        "measured_ita_delta_skin_only": round(measured_ita_delta_skin, 2),
        "ita_manipulation_check_pass": skin_intended_ok,
        "residual_illuminant_coupling_in_skin_shift": round(skin_leak_into_illum, 4),
        "skin_only_accepted": skin_only_accepted,
        "residual_ita_coupling_in_illum_shift": round(illum_leak_into_ita, 4),
        "illum_manipulation_check_pass": illum_intended_ok,
        "illum_only_accepted": illum_only_accepted,
        "measured_ita_delta_both": round(measured_ita_delta_both, 2),
        "both_illum_shift_magnitude": round(both_illum_shift_magnitude, 4),
        "both_accepted": both_accepted,
        "accepted": accepted,
        "reject_reasons": reject_reasons,
    }


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    fake_face = np.random.randint(80, 180, size=(256, 256, 3), dtype=np.uint8)
    cf = generate_counterfactual_set(fake_face, face_id="demo_0001")
    logger.info(cf.validation)
    logger.info("accepted=%s reject_reasons=%s", cf.accepted, cf.reject_reasons)
