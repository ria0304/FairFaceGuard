"""Celeb-DF (v2) dataset adapter -- cross-dataset validation set.

Celeb-DF is never used for training in this pipeline. Its only role is
`src/experiments/cross_dataset_validator.py`: train the baseline on FF++,
then evaluate zero-shot on Celeb-DF to check whether the skin-tone/
illumination subgroup gaps found on FF++ generalize to a completely
different manipulation method (Celeb-DF uses a different, higher-quality
GAN-based face-swap pipeline than any FF++ manipulation) and a different
pool of source identities. See `DATASET_SELECTION.md`.

Produces the same labels.csv + annotations.csv contract as
`ffplus_adapter.py`, via the same frame-extraction + Week 1 ITA-annotation
path. Because this set is only ever used for held-out evaluation, every row
is written with fake_label populated but split left unset by default --
callers should route ALL of it to "test" (there is no train/val use for
this dataset; see `identity_split_celebdf` below, which is a thin wrapper
that assigns everything to "test" while still deduplicating by identity for
reporting purposes).

--------------------------------------------------------------------
Celeb-DF (v2) directory layout assumed (matches the official release):
--------------------------------------------------------------------
    celebdf_root/
        Celeb-real/
            <id>_<clip>.mp4              # real videos of real celebrities
        YouTube-real/
            <id>.mp4                     # real videos, YouTube source
        Celeb-synthesis/
            <id>_<target_id>_<clip>.mp4  # synthesized (fake) videos
        List_of_testing_videos.txt        # official train/test protocol (optional)

Identity key: the leading numeric celebrity id in the filename
(e.g. 'id0_0000.mp4' -> 'id0'; 'id0_id5_0003.mp4' -> the SOURCE identity
id0). Kept in the same '<id>__f<frame>_<label>' face_id shape as
ffplus_adapter.py for identity_split.py compatibility, even though every
row from this file will be routed to the test split.

Usage:
    python -m src.data.celebdf_adapter \\
        --celebdf_root /data/Celeb-DF \\
        --output_dir /data/celebdf_processed \\
        --frames_per_video 10 \\
        --sample_n 200          # optional: subsample videos for a fast first pass
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.annotation.ita_fitzpatrick import annotate_face

REAL_SUBDIRS = ["Celeb-real", "YouTube-real"]
FAKE_SUBDIR = "Celeb-synthesis"


@dataclass
class CelebDFStats:
    total_videos: int = 0
    frames_extracted: int = 0
    faces_detected: int = 0
    annotation_failed: int = 0


def parse_celebdf_identity(filename: str) -> str:
    """'id0_0000.mp4' -> 'id0'; 'id0_id5_0003.mp4' -> 'id0' (source identity);
    bare 'id12.mp4' (YouTube-real sometimes has no clip suffix) -> 'id12'."""
    stem = Path(filename).stem
    parts = stem.split("_")
    return parts[0]


def build_celebdf_metadata(celebdf_root: str) -> pd.DataFrame:
    """Build a DataFrame of all video paths with fake_label and identity keys."""
    rows = []

    for subdir in REAL_SUBDIRS:
        real_dir = os.path.join(celebdf_root, subdir)
        if not os.path.exists(real_dir):
            continue
        for vid in sorted(os.listdir(real_dir)):
            if vid.endswith((".mp4", ".avi")):
                rows.append({
                    "video_path": os.path.join(real_dir, vid),
                    "video_id": parse_celebdf_identity(vid),
                    "fake_label": 0,
                    "source_subdir": subdir,
                })

    fake_dir = os.path.join(celebdf_root, FAKE_SUBDIR)
    if os.path.exists(fake_dir):
        for vid in sorted(os.listdir(fake_dir)):
            if vid.endswith((".mp4", ".avi")):
                rows.append({
                    "video_path": os.path.join(fake_dir, vid),
                    "video_id": parse_celebdf_identity(vid),
                    "fake_label": 1,
                    "source_subdir": FAKE_SUBDIR,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No videos found under {celebdf_root}. Expected "
            f"'{'/ or /'.join(REAL_SUBDIRS)}/' and '{FAKE_SUBDIR}/'. "
            "Check --celebdf_root."
        )
    return df


def convert_celebdf(
    celebdf_root: str,
    output_dir: str,
    frames_per_video: int = 10,
    face_size: int = 380,
    sample_n: int | None = None,
    seed: int = 42,
) -> CelebDFStats:
    """Extract face crops from Celeb-DF videos, annotate with Week 1's ITA
    pipeline, and write labels.csv + annotations.csv with every row's
    'split' column set to 'test' -- this dataset is held out only, never
    trained on."""
    from src.data.frame_extractor import FaceExtractor, extract_face_crops_from_video

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    stats = CelebDFStats()
    df = build_celebdf_metadata(celebdf_root)
    stats.total_videos = len(df)

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    extractor = FaceExtractor(image_size=face_size)
    label_rows = []
    annotation_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Celeb-DF videos"):
        video_path = row["video_path"]
        video_id = row["video_id"]
        fake_label = row["fake_label"]

        crops = extract_face_crops_from_video(
            video_path, max_frames=frames_per_video, target_size=face_size, extractor=extractor
        )
        stats.frames_extracted += len(crops)

        for frame_idx, face_bgr in enumerate(crops):
            face_id = f"{video_id}__f{frame_idx:03d}_{'fake' if fake_label else 'real'}"
            frame_path = os.path.join(frames_dir, f"{face_id}.png")
            cv2.imwrite(frame_path, face_bgr)

            result = annotate_face(face_bgr, face_id=face_id)

            label_rows.append({
                "face_id": face_id,
                "fake_label": fake_label,
                "video_id": video_id,
                "frame_idx": frame_idx,
                "split": "test",  # Celeb-DF is held-out only; never trained on
            })

            annotation_rows.append({
                "face_id": face_id,
                "ita_continuous": result.ita_continuous,
                "fitzpatrick_bin": None if result.fitzpatrick_bin == "unknown" else result.fitzpatrick_bin,
                "fitzpatrick_bin_source": "auto_ita",
                "illuminant_estimate_r": result.illuminant_estimate[2],
                "illuminant_estimate_g": result.illuminant_estimate[1],
                "illuminant_estimate_b": result.illuminant_estimate[0],
                "patch_confidence": result.patch_confidence,
                "flagged": bool(result.flagged or result.fitzpatrick_bin == "unknown"),
                "video_id": video_id,
            })

            if result.fitzpatrick_bin == "unknown":
                stats.annotation_failed += 1
            else:
                stats.faces_detected += 1

    labels_df = pd.DataFrame(label_rows)
    annotations_df = pd.DataFrame(annotation_rows)

    labels_path = os.path.join(output_dir, "labels.csv")
    annotations_path = os.path.join(output_dir, "annotations.csv")
    labels_df.to_csv(labels_path, index=False)
    annotations_df.to_csv(annotations_path, index=False)

    print(f"[celebdf_adapter] videos: {stats.total_videos}, frames: {stats.frames_extracted}, "
          f"faces OK: {stats.faces_detected}, failed: {stats.annotation_failed}")
    print(f"[celebdf_adapter] wrote {len(labels_df)} rows -> {labels_path}")
    print(f"[celebdf_adapter] wrote {len(annotations_df)} rows -> {annotations_path}")

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--celebdf_root", required=True, help="Root of the Celeb-DF (v2) download")
    p.add_argument("--output_dir", required=True, help="data_root to write frames/ + labels.csv/annotations.csv into")
    p.add_argument("--frames_per_video", type=int, default=10)
    p.add_argument("--face_size", type=int, default=380)
    p.add_argument("--sample_n", type=int, default=None, help="Subsample N videos for a fast first pass")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    convert_celebdf(
        celebdf_root=args.celebdf_root,
        output_dir=args.output_dir,
        frames_per_video=args.frames_per_video,
        face_size=args.face_size,
        sample_n=args.sample_n,
        seed=args.seed,
    )


if __name__ == "__main__":
    # Metadata-building smoke test (no torch/facenet-pytorch or real
    # download needed) -- mirrors ffplus_adapter.py's smoke test.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for subdir in REAL_SUBDIRS:
            d = os.path.join(tmp, subdir)
            os.makedirs(d, exist_ok=True)
        open(os.path.join(tmp, "Celeb-real", "id0_0000.mp4"), "w").close()
        open(os.path.join(tmp, "Celeb-real", "id1_0000.mp4"), "w").close()
        open(os.path.join(tmp, "YouTube-real", "id2.mp4"), "w").close()

        fake_dir = os.path.join(tmp, FAKE_SUBDIR)
        os.makedirs(fake_dir, exist_ok=True)
        open(os.path.join(fake_dir, "id0_id1_0000.mp4"), "w").close()

        df = build_celebdf_metadata(tmp)
        assert len(df) == 4, f"expected 4 rows, got {len(df)}"
        assert set(df[df["fake_label"] == 0]["video_id"]) == {"id0", "id1", "id2"}
        assert set(df[df["fake_label"] == 1]["video_id"]) == {"id0"}
        assert parse_celebdf_identity("id0_id5_0003.mp4") == "id0"

        print("[smoke test] build_celebdf_metadata produced:")
        print(df)
        print("[smoke test] PASSED")
