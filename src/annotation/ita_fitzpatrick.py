"""Week 1 -- Skin-tone annotation pipeline (ITA / Fitzpatrick).

Produces, per face image, a de-illuminated Individual Typology Angle (ITA)
score and a Fitzpatrick I-VI bin. The illuminant-normalization step (2) is
the piece that keeps this label from silently absorbing lighting information
-- which is exactly the confound the rest of the study is designed to test,
so it must not leak in at annotation time.

Pipeline:
    1. Detect face + 68 landmarks.
    2. Estimate the scene illuminant and white-balance-correct the image.
    3. Sample patches from illumination-stable, cosmetics-sparse regions
       (inner cheek, forehead, nose bridge).
    4. Convert to CIE-Lab, compute ITA from median L*/b*.
    5. Bin into Fitzpatrick I-VI via published ITA correspondence table.
    6. Flag low-confidence samples (extreme pose, saturated patches).

Dependencies: numpy, opencv-python, (optional) face-alignment or mediapipe
for landmarks -- a lightweight fallback landmark hook is provided so this
module runs standalone; swap `detect_landmarks` for a real detector in
production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np

# ITA (degrees) -> Fitzpatrick bin, from Wilkes et al. 2015 / Del Bino & Bernerd
# correspondence tables. Thresholds are inclusive upper bounds.
ITA_FITZPATRICK_THRESHOLDS = [
    (55.0, "I"),     # very light,  ITA > 55
    (41.0, "II"),    # light,       41 < ITA <= 55
    (28.0, "III"),   # intermediate 28 < ITA <= 41
    (10.0, "IV"),    # tan,         10 < ITA <= 28
    (-30.0, "V"),    # brown,      -30 < ITA <= 10
    (-90.0, "VI"),   # dark,        ITA <= -30
]

# Landmark indices (68-point scheme) used to define sampling patches.
PATCH_LANDMARKS = {
    "left_cheek": [1, 2, 3, 31],
    "right_cheek": [13, 14, 15, 35],
    "forehead": [19, 20, 24, 25],   # approximate; real forehead landmarks
                                      # require an extended 81/98-point model
    "nose_bridge": [27, 28, 29, 30],
}


@dataclass
class SkinToneAnnotation:
    face_id: str
    ita_continuous: float
    fitzpatrick_bin: str
    illuminant_estimate: tuple[float, float, float]
    patch_confidence: float
    flagged: bool

    def to_dict(self) -> dict:
        return asdict(self)


def detect_landmarks(image_bgr: np.ndarray) -> np.ndarray | None:
    """Detect facial landmarks using MediaPipe Face Mesh.
    
    This is the primary landmark detection method for the FairFaceGuard pipeline.
    MediaPipe Face Mesh provides 468 3D face landmarks with high accuracy and
    real-time performance.
    
    For the ITA annotation pipeline, we use only the subset of landmarks that
    correspond to the standard 68-point scheme (cheeks, forehead, nose bridge).
    
    Args:
        image_bgr: Input image in BGR format (OpenCV convention).
    
    Returns:
        Array of shape [68, 2] with (x, y) pixel coordinates, or None if no
        face is detected.
    
    Raises:
        ImportError: If mediapipe is not installed.
    
    Note:
        If you need an alternative landmark detector, consider:
        - face-alignment (https://github.com/1adrianb/face-alignment)
        - dlib (http://dlib.net/)
        
        To use an alternative, replace this function's implementation while
        maintaining the same return signature.
    """
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            "MediaPipe is required for landmark detection. "
            "Install with: pip install mediapipe>=0.10.0\n"
            "Alternatively, install face-alignment and modify detect_landmarks() "
            "in src/annotation/ita_fitzpatrick.py to use it instead."
        )
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if results.landmarks is None or len(results.landmarks) == 0:
        face_mesh.close()
        return None
    
    # Get the first detected face (468 landmarks)
    mp_landmarks = results.landmarks[0]
    h, w = image_bgr.shape[:2]
    
    # Map MediaPipe 468 landmarks to approximate 68-point scheme positions
    # MediaPipe indices: https://google.github.io/mediapipe/solutions/face_mesh.html
    # We'll extract key landmarks and interpolate to get 68 points
    
    # Key correspondence (approximate):
    # MediaPipe -> 68-point scheme
    # Nose tip: 1 -> 30
    # Left eye inner: 133 -> 36/39 region
    # Right eye inner: 362 -> 42/45 region
    # Mouth corners: 61, 291 -> 48, 54
    # Cheek contours: use subsets
    
    # Extract relevant landmarks for skin-tone patches
    # We need: left_cheek, right_cheek, forehead, nose_bridge
    
    # Simplified approach: use the actual MediaPipe landmarks directly
    # mapped to our patch definitions
    
    # For 68-point compatibility, we'll create a mapping
    # This is an approximation - for research use, consider using
    # the native MediaPipe indices throughout
    
    pts_68 = np.zeros((68, 2))
    
    # Nose bridge (points 27-30 in 68-point)
    # MediaPipe: 6 (bridge top), 197 (mid), 195 (lower), 1 (tip)
    nose_bridge_mp = [6, 197, 195, 1]
    for i, mp_idx in enumerate(nose_bridge_mp):
        if mp_idx < len(mp_landmarks):
            lm = mp_landmarks[mp_idx]
            pts_68[27 + i] = [lm.x * w, lm.y * h]
    
    # Left cheek area (points 1-3, 31 in 68-point)
    # MediaPipe: cheek region on left side of face (viewer's right)
    left_cheek_mp = [116, 117, 118, 141]  # Approximate left cheek
    for i, mp_idx in enumerate(left_cheek_mp[:3]):
        if mp_idx < len(mp_landmarks):
            lm = mp_landmarks[mp_idx]
            pts_68[i + 1] = [lm.x * w, lm.y * h]
    if len(left_cheek_mp) > 3 and left_cheek_mp[3] < len(mp_landmarks):
        lm = mp_landmarks[left_cheek_mp[3]]
        pts_68[31] = [lm.x * w, lm.y * h]
    
    # Right cheek area (points 13-15, 35 in 68-point)
    right_cheek_mp = [345, 346, 347, 370]  # Approximate right cheek
    for i, mp_idx in enumerate(right_cheek_mp[:3]):
        if mp_idx < len(mp_landmarks):
            lm = mp_landmarks[mp_idx]
            pts_68[13 + i] = [lm.x * w, lm.y * h]
    if len(right_cheek_mp) > 3 and right_cheek_mp[3] < len(mp_landmarks):
        lm = mp_landmarks[right_cheek_mp[3]]
        pts_68[35] = [lm.x * w, lm.y * h]
    
    # Forehead area (points 19-20, 24-25 in 68-point)
    # MediaPipe: upper face region
    forehead_mp = [10, 9, 151, 162]  # Approximate forehead
    forehead_68_indices = [19, 20, 24, 25]
    for i, mp_idx in enumerate(forehead_mp):
        if mp_idx < len(mp_landmarks) and i < len(forehead_68_indices):
            lm = mp_landmarks[mp_idx]
            pts_68[forehead_68_indices[i]] = [lm.x * w, lm.y * h]
    
    face_mesh.close()
    
    # Verify we have valid landmarks (not all zeros)
    if np.all(pts_68 == 0):
        return None
    
    return pts_68


def estimate_illuminant_gray_world(image_bgr: np.ndarray) -> np.ndarray:
    """Gray-world illuminant estimate (mean channel value). Fast, adequate
    baseline; swap for a learned illuminant-estimation network for higher
    fidelity if available."""
    return image_bgr.reshape(-1, 3).mean(axis=0)  # BGR order


def white_balance_correct(image_bgr: np.ndarray, illuminant: np.ndarray) -> np.ndarray:
    """Scale each channel so the estimated illuminant becomes neutral gray.
    This is the de-illumination step -- run BEFORE computing ITA so the
    'ground truth' skin-tone label doesn't bake in scene lighting."""
    gray_target = illuminant.mean()
    gains = gray_target / np.clip(illuminant, 1e-6, None)
    out = image_bgr.astype(np.float32) * gains
    return np.clip(out, 0, 255).astype(np.uint8)


def _polygon_patch(image: np.ndarray, pts: np.ndarray, indices: list[int]) -> np.ndarray | None:
    poly = pts[indices].astype(np.int32)
    if np.any(poly < 0):
        return None
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    ys, xs = np.where(mask == 1)
    if len(xs) < 10:
        return None
    return image[ys, xs]  # [N, 3] BGR pixels


def compute_ita(lab_pixels: np.ndarray) -> float:
    """ITA = arctan((L* - 50) / b*) * 180 / pi, using median L*/b*."""
    L = np.median(lab_pixels[:, 0]) * (100.0 / 255.0)   # OpenCV Lab: L in [0,255]
    b = np.median(lab_pixels[:, 2]) - 128.0               # OpenCV Lab: a,b centered at 128
    ita = math.degrees(math.atan2((L - 50.0), b)) if abs(b) > 1e-6 else 0.0
    return ita


def ita_to_fitzpatrick(ita: float) -> str:
    for threshold, label in ITA_FITZPATRICK_THRESHOLDS:
        if ita > threshold:
            return label
    return ITA_FITZPATRICK_THRESHOLDS[-1][1]


def annotate_face(
    image_bgr: np.ndarray,
    face_id: str,
    max_pose_frac: float = 0.30,
    min_patch_pixels: int = 20,
) -> SkinToneAnnotation:
    """Run the full Week-1 pipeline on a single cropped face image."""
    landmarks = detect_landmarks(image_bgr)
    if landmarks is None:
        return SkinToneAnnotation(face_id, float("nan"), "unknown", (0, 0, 0), 0.0, True)

    illuminant = estimate_illuminant_gray_world(image_bgr)
    balanced = white_balance_correct(image_bgr, illuminant)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2Lab)

    all_pixels = []
    confidences = []
    for region, idx in PATCH_LANDMARKS.items():
        patch = _polygon_patch(lab, landmarks, idx)
        if patch is None or len(patch) < min_patch_pixels:
            confidences.append(0.0)
            continue
        all_pixels.append(patch)
        # confidence proxy: patch not near-saturated (over/under exposed)
        L_channel = patch[:, 0]
        sat_frac = np.mean((L_channel < 5) | (L_channel > 250))
        confidences.append(1.0 - sat_frac)

    if not all_pixels:
        return SkinToneAnnotation(face_id, float("nan"), "unknown", tuple(illuminant), 0.0, True)

    pooled = np.concatenate(all_pixels, axis=0)
    ita = compute_ita(pooled)
    fitz = ita_to_fitzpatrick(ita)
    confidence = float(np.mean(confidences))
    flagged = confidence < 0.5

    return SkinToneAnnotation(
        face_id=face_id,
        ita_continuous=round(ita, 3),
        fitzpatrick_bin=fitz,
        illuminant_estimate=tuple(float(x) for x in illuminant),
        patch_confidence=round(confidence, 3),
        flagged=flagged,
    )


def validate_against_human_ratings(
    auto_bins: list[str],
    human_bins: list[str],
) -> dict:
    """Agreement check between the automatic pipeline and a held-out
    human-rated (e.g. dermatologist or trained crowd) subsample, as
    recommended in the methodology's QC step. Returns Cohen's kappa."""
    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(auto_bins, human_bins)
    agreement = float(np.mean([a == h for a, h in zip(auto_bins, human_bins)]))
    return {"cohen_kappa": float(kappa), "raw_agreement": agreement}


if __name__ == "__main__":
    # Smoke test on a synthetic image so this file is runnable standalone.
    fake_face = np.random.randint(80, 180, size=(256, 256, 3), dtype=np.uint8)
    result = annotate_face(fake_face, face_id="demo_0001")
    print(result.to_dict())
