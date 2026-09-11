"""AI-Face dataset adapter.

Converts Purdue-M2's AI-Face `train.csv` / `test.csv` (the real,
published schema -- see https://github.com/Purdue-M2/AI-Face-FairnessBench)
into the `labels.csv` + `annotations.csv` layout that
`src/data/datasets.py` already expects, so the rest of this pipeline
(Weeks 1-5) does not need to change at all.

Why this file exists instead of editing datasets.py directly:
datasets.py documents itself as "the only place other modules assume a
specific directory structure" -- this adapter is a one-way translator
that runs BEFORE datasets.py ever sees the data, so that contract stays
true.

--------------------------------------------------------------------
AI-Face ships TWO annotation schema versions. Detect and handle both.
--------------------------------------------------------------------

v1 columns:
    Image Path, Gender, Age, Skin Tone, Intersection, Target
    -- "Skin Tone" is on the Monk Skin Tone Scale (10-point, MST-1..MST-10
       or sometimes given as a bare int 1-10 / 0-9 depending on release).
       This is a genuine skin-tone label and is what you want if you can
       get v1 annotations.

v2 columns (current/default download):
    Image Path, Uncertainty Score Gender, Uncertainty Score Age,
    Uncertainty Score Race, Ground Truth Gender, Ground Truth Age,
    Ground Truth Race, Intersection, Target
    -- v2 DROPPED the direct skin-tone column and replaced it with
       "Race" (0=Asian, 1=White, 2=Black, 3=Others). This is a coarser,
       *categorical* proxy for skin tone, not a skin-tone scale. Treat
       any Fitzpatrick bin derived from it as a weak proxy label, and
       prefer your own ITA pipeline's auto-derived bin as the primary
       signal wherever the two disagree. This module writes BOTH so you
       can compare them (see `fitzpatrick_bin_source` column below).

Illumination: AI-Face provides no illumination label at all in either
version. `annotations.csv`'s illuminant_estimate_{r,g,b} columns are
always produced by actually running Week 1's `estimate_illuminant_gray_world`
on the real pixels -- there's no shortcut around opening each image.

Usage:
    python -m src.data.aiface_adapter \\
        --aiface_csv /path/to/train.csv \\
        --image_root /path/to/AI-Face/Dataset/root \\
        --output_dir /path/to/data_root \\
        --sample_n 2000          # optional: subsample for a fast first pass
        --split train
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from src.annotation.ita_fitzpatrick import (
    annotate_face,
    estimate_illuminant_gray_world,
)

# --------------------------------------------------------------------
# Schema detection & column normalization
# --------------------------------------------------------------------

V1_SKIN_COL = "Skin Tone"
V2_RACE_COL = "Ground Truth Race"

# Monk Skin Tone Scale (1-10, light -> dark) collapsed into Fitzpatrick
# I-VI. This mapping is a coarse, documented approximation -- there is
# no official MST<->Fitzpatrick correspondence table, so state this
# explicitly wherever fitzpatrick_bin_source == "monk_proxy" is used in
# the paper.
MONK_TO_FITZPATRICK = {
    1: "I", 2: "I",
    3: "II", 4: "II",
    5: "III", 6: "III",
    7: "IV",
    8: "V",
    9: "VI", 10: "VI",
}

# v2's coarse Race category collapsed into Fitzpatrick. This is a WEAK
# proxy (race != skin tone) -- use only as a fallback when no MST label
# and no auto-ITA bin is available, and always report results computed
# from the auto-ITA bin as primary.
RACE_TO_FITZPATRICK_PROXY = {
    0: "III",  # Asian -> mid-range proxy, highly heterogeneous in reality
    1: "II",   # White -> light proxy
    2: "V",    # Black -> dark proxy
    3: "III",  # Others -> unknown, mid-range default
}


def detect_schema_version(df: pd.DataFrame) -> str:
    if V1_SKIN_COL in df.columns:
        return "v1"
    if V2_RACE_COL in df.columns:
        return "v2"
    raise ValueError(
        f"Unrecognized AI-Face CSV schema. Columns found: {list(df.columns)}. "
        "Expected either a 'Skin Tone' column (v1) or 'Ground Truth Race' "
        "column (v2). Check you're pointing at the actual train.csv/test.csv "
        "from https://github.com/Purdue-M2/AI-Face-FairnessBench."
    )


def _parse_monk_value(raw) -> int | None:
    """Handles 'MST-7', '7', 7, 7.0, etc."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().upper().replace("MST-", "").replace("MST", "")
    try:
        val = int(float(s))
    except ValueError:
        return None
    return val if 1 <= val <= 10 else None


def face_id_from_path(image_path: str) -> str:
    """Deterministic, filesystem-safe face_id derived from AI-Face's
    Image Path column (which includes subset/method subdirectories, so
    plain basenames can collide across subsets -- keep the path info)."""
    cleaned = image_path.strip().replace("\\", "/").lstrip("/")
    stem, _ext = os.path.splitext(cleaned)
    return stem.replace("/", "__")


# --------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------

@dataclass
class ConversionStats:
    total_rows: int = 0
    processed: int = 0
    missing_image: int = 0
    landmark_failed: int = 0
    monk_proxy_used: int = 0
    race_proxy_used: int = 0
    auto_ita_used: int = 0


def convert_aiface(
    aiface_csv: str,
    image_root: str,
    output_dir: str,
    sample_n: int | None = None,
    split_name: str | None = None,
    seed: int = 42,
    run_illuminant_estimation: bool = True,
) -> ConversionStats:
    """Reads an AI-Face train.csv/test.csv, opens each image once to run
    the Week 1 illuminant estimator (illumination has no ground-truth
    label in AI-Face, so this step is not optional if you want illum_bin
    populated), and writes labels.csv + annotations.csv in the schema
    src/data/datasets.py expects.
    """
    os.makedirs(output_dir, exist_ok=True)
    stats = ConversionStats()

    df = pd.read_csv(aiface_csv)
    stats.total_rows = len(df)
    schema = detect_schema_version(df)

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    label_rows = []
    annotation_rows = []

    for _, row in df.iterrows():
        image_path = row["Image Path"]
        face_id = face_id_from_path(image_path)
        fake_label = int(row["Target"])

        label_rows.append({
            "face_id": face_id,
            "fake_label": fake_label,
            **({"split": split_name} if split_name else {}),
        })

        # --- Ground-truth-derived Fitzpatrick proxy (from AI-Face's own labels) ---
        ground_truth_bin = None
        bin_source = None
        if schema == "v1":
            monk_val = _parse_monk_value(row.get(V1_SKIN_COL))
            if monk_val is not None:
                ground_truth_bin = MONK_TO_FITZPATRICK[monk_val]
                bin_source = "monk_proxy"
                stats.monk_proxy_used += 1
        else:  # v2
            race_val = row.get(V2_RACE_COL)
            if pd.notna(race_val):
                ground_truth_bin = RACE_TO_FITZPATRICK_PROXY.get(int(race_val))
                bin_source = "race_proxy"
                stats.race_proxy_used += 1

        # --- Auto ITA / illuminant estimate from the real pixels (Week 1) ---
        full_path = os.path.join(image_root, image_path)
        ita_continuous = float("nan")
        auto_fitz = None
        illuminant = (float("nan"),) * 3
        confidence = 0.0
        flagged = True

        if run_illuminant_estimation:
            if not os.path.exists(full_path):
                stats.missing_image += 1
                flagged = True
            else:
                image_bgr = cv2.imread(full_path)
                if image_bgr is None:
                    stats.missing_image += 1
                else:
                    result = annotate_face(image_bgr, face_id=face_id)
                    ita_continuous = result.ita_continuous
                    auto_fitz = None if result.fitzpatrick_bin == "unknown" else result.fitzpatrick_bin
                    illuminant = result.illuminant_estimate
                    confidence = result.patch_confidence
                    flagged = result.flagged
                    if auto_fitz is not None:
                        stats.auto_ita_used += 1
                    else:
                        stats.landmark_failed += 1

        # Primary bin: prefer auto-ITA (illumination-corrected, continuous,
        # produced by YOUR pipeline) over AI-Face's own coarse ground-truth
        # proxy. Fall back to the ground-truth proxy only if auto-ITA failed.
        final_bin = auto_fitz if auto_fitz is not None else ground_truth_bin
        final_source = "auto_ita" if auto_fitz is not None else bin_source

        annotation_rows.append({
            "face_id": face_id,
            "ita_continuous": ita_continuous,
            "fitzpatrick_bin": final_bin,
            "fitzpatrick_bin_source": final_source,   # "auto_ita" | "monk_proxy" | "race_proxy" | None
            "ground_truth_fitzpatrick_proxy": ground_truth_bin,
            "illuminant_estimate_r": illuminant[2] if illuminant else float("nan"),  # cv2 is BGR
            "illuminant_estimate_g": illuminant[1] if illuminant else float("nan"),
            "illuminant_estimate_b": illuminant[0] if illuminant else float("nan"),
            "patch_confidence": confidence,
            "flagged": bool(flagged) or final_bin is None,
        })
        stats.processed += 1

    labels_df = pd.DataFrame(label_rows)
    annotations_df = pd.DataFrame(annotation_rows)

    labels_path = os.path.join(output_dir, "labels.csv")
    annotations_path = os.path.join(output_dir, "annotations.csv")
    labels_df.to_csv(labels_path, index=False)
    annotations_df.to_csv(annotations_path, index=False)

    print(f"[aiface_adapter] schema detected: {schema}")
    print(f"[aiface_adapter] wrote {len(labels_df)} rows -> {labels_path}")
    print(f"[aiface_adapter] wrote {len(annotations_df)} rows -> {annotations_path}")
    print(f"[aiface_adapter] stats: {stats}")

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aiface_csv", required=True, help="Path to AI-Face train.csv or test.csv")
    p.add_argument("--image_root", required=True, help="Root directory the 'Image Path' column is relative to")
    p.add_argument("--output_dir", required=True, help="data_root to write labels.csv/annotations.csv into")
    p.add_argument("--sample_n", type=int, default=None, help="Subsample N rows for a fast first pass (recommended before a full run)")
    p.add_argument("--split", default=None, help="Optional split label ('train'/'test') written into labels.csv")
    p.add_argument("--skip_illuminant", action="store_true", help="Skip opening images / illuminant estimation (fast, but illum_bin will be unusable)")
    return p


def main():
    args = build_arg_parser().parse_args()
    convert_aiface(
        aiface_csv=args.aiface_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        sample_n=args.sample_n,
        split_name=args.split,
        run_illuminant_estimation=not args.skip_illuminant,
    )


if __name__ == "__main__":
    # Smoke test on a synthetic AI-Face-shaped CSV + synthetic images,
    # so this is runnable/testable without the real (multi-GB) dataset.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        image_root = os.path.join(tmp, "images")
        os.makedirs(os.path.join(image_root, "Real", "FFHQ"), exist_ok=True)
        os.makedirs(os.path.join(image_root, "GANs", "StarGAN"), exist_ok=True)

        rows = []
        for i in range(6):
            is_fake = i % 2
            subdir = "GANs/StarGAN" if is_fake else "Real/FFHQ"
            fname = f"img_{i:03d}.png"
            rel_path = f"{subdir}/{fname}"
            abs_path = os.path.join(image_root, subdir, fname)
            synthetic = np.random.randint(60, 200, size=(256, 256, 3), dtype=np.uint8)
            cv2.imwrite(abs_path, synthetic)
            rows.append({
                "Image Path": rel_path,
                "Uncertainty Score Gender": 0.1,
                "Uncertainty Score Age": 0.1,
                "Uncertainty Score Race": 0.1,
                "Ground Truth Gender": i % 2,
                "Ground Truth Age": 1,
                "Ground Truth Race": i % 4,
                "Intersection": i % 8,
                "Target": is_fake,
            })

        csv_path = os.path.join(tmp, "train.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        out_dir = os.path.join(tmp, "data_root")
        stats = convert_aiface(
            aiface_csv=csv_path,
            image_root=image_root,
            output_dir=out_dir,
            split_name="train",
        )
        print("\n[smoke test] labels.csv:")
        print(pd.read_csv(os.path.join(out_dir, "labels.csv")))
        print("\n[smoke test] annotations.csv:")
        print(pd.read_csv(os.path.join(out_dir, "annotations.csv")))
        assert stats.processed == 6
        print("\n[smoke test] PASSED")
