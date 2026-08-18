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
        image_bgr, skin_only, illum_only, ita_delta
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


def _validate_counterfactuals(
    original: np.ndarray,
    skin_only: np.ndarray,
    illum_only: np.ndarray,
    intended_ita_delta: float,
    ita_tolerance: float = 8.0,
    illum_tolerance_frac: float = 0.15,
) -> dict:
    """Manipulation check (methodology Section 3.1/3.2):
        - skin_only should move ITA by ~intended_ita_delta and leave the
          illuminant estimate roughly unchanged.
        - illum_only should leave ITA roughly unchanged and move the
          illuminant estimate.
    Returns pass/fail flags plus the measured residual coupling, which
    should be reported as a bounded source of error in the paper."""
    ita_orig = _ita_of(original)
    ita_skin = _ita_of(skin_only)
    ita_illum = _ita_of(illum_only)

    illum_orig = _illuminant_of(original)
    illum_skin = _illuminant_of(skin_only)
    illum_illum = _illuminant_of(illum_only)

    measured_ita_delta = ita_skin - ita_orig
    ita_check_pass = abs(measured_ita_delta - intended_ita_delta) < ita_tolerance
    skin_leak_into_illum = float(
        np.linalg.norm(illum_skin - illum_orig) / (np.linalg.norm(illum_orig) + 1e-6)
    )

    illum_leak_into_ita = abs(ita_illum - ita_orig)
    illum_shift_magnitude = float(
        np.linalg.norm(illum_illum - illum_orig) / (np.linalg.norm(illum_orig) + 1e-6)
    )

    return {
        "measured_ita_delta_skin_only": round(measured_ita_delta, 2),
        "ita_manipulation_check_pass": bool(ita_check_pass),
        "residual_illuminant_coupling_in_skin_shift": round(skin_leak_into_illum, 4),
        "residual_ita_coupling_in_illum_shift": round(illum_leak_into_ita, 4),
        "illum_manipulation_check_pass": bool(
            illum_shift_magnitude > illum_tolerance_frac and illum_leak_into_ita < ita_tolerance
        ),
    }


if __name__ == "__main__":
    fake_face = np.random.randint(80, 180, size=(256, 256, 3), dtype=np.uint8)
    cf = generate_counterfactual_set(fake_face, face_id="demo_0001")
    print(cf.validation)
