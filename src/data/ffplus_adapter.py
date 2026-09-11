"""FaceForensics++ (FF++) dataset adapter -- primary real-video dataset.

Replaces `aiface_adapter.py` as the primary data source (see
`DATASET_SELECTION.md` for the rationale). Converts FF++'s raw video
layout into the `labels.csv` + `annotations.csv` layout
`src/data/datasets.py` expects, the same contract `aiface_adapter.py`
honors -- so nothing in Weeks 3-5 needs to change.

Unlike AI-Face (pre-cropped stills, no illumination ground truth, no
identity metadata), FF++ requires one extra step first: extracting face
crops from video (`src/data/frame_extractor.py`). Everything after that
-- ITA/Fitzpatrick annotation, illuminant estimation -- reuses Week 1's
`annotate_face` unchanged, run on the real extracted pixels.

FF++ ships no skin-tone or race label at all (an improvement over
AI-Face's coarse race-proxy: this dataset does not tempt you to conflate
race with skin tone). The auto-ITA bin from `annotate_face` is therefore
the ONLY skin-tone signal here, not a fallback -- there is no ground-truth
proxy to write into a `fitzpatrick_bin_source` alternate column the way
aiface_adapter.py does.

--------------------------------------------------------------------
FF++ directory layout assumed (matches the official release):
--------------------------------------------------------------------
    ffpp_root/
        original_sequences/
            youtube/
                c23/                      # or c40, raw
                    videos/
                        <video_id>.mp4
        manipulated_sequences/
            Deepfakes/
                c23/
                    videos/
                        <video_id>_<target_id>.mp4
            Face2Face/
                c23/videos/...
            FaceSwap/
                c23/videos/...
            NeuralTextures/
                c23/videos/...

Identity key: the SOURCE video_id (the id before the underscore in a
manipulated filename, or the bare filename for originals). Every frame
extracted from every manipulation of the same source actor gets the same
`identity_key`-compatible face_id prefix, so `src/data/identity_split.py`
(group_by="face_id_prefix" or "source_video") keeps that actor entirely
within one split -- this is the actual fix for the leakage risk FF++
introduces that AI-Face never had (AI-Face has no video/identity
structure to leak across).

Usage:
    python -m src.data.ffplus_adapter \\
        --ffpp_root /data/ff++ \\
        --output_dir /data/ffplus_processed \\
        --manipulation all \\
        --compression c23 \\
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

MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


@dataclass
class FFPlusStats:
    total_videos: int = 0
    frames_extracted: int = 0
    faces_detected: int = 0
    annotation_failed: int = 0


def parse_ffplus_video_id(filename: str) -> str:
    """Deepfakes/Face2Face/FaceSwap/NeuralTextures naming is
    '<source>_<target>.mp4' -> identity = source (the real actor whose
    face is being swapped/reenacted). Originals are '<source>.mp4' ->
    identity = source directly."""
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def build_ffplus_metadata(
    ffpp_root: str,
    manipulation: str = "all",
    compression: str = "c23",
) -> pd.DataFrame:
    """Build a DataFrame of all video paths with fake_label and identity keys."""
    rows = []
    originals_dir = os.path.join(ffpp_root, "original_sequences", "youtube", compression, "videos")

    if os.path.exists(originals_dir):
        for vid in sorted(os.listdir(originals_dir)):
            if vid.endswith((".mp4", ".avi")):
                rows.append({
                    "video_path": os.path.join(originals_dir, vid),
                    "video_id": parse_ffplus_video_id(vid),
                    "fake_label": 0,
                    "manipulation": "original",
                })

    manip_types = MANIPULATION_TYPES if manipulation == "all" else [manipulation]
    for manip in manip_types:
        manip_dir = os.path.join(ffpp_root, "manipulated_sequences", manip, compression, "videos")
        if not os.path.exists(manip_dir):
            continue
        for vid in sorted(os.listdir(manip_dir)):
            if vid.endswith((".mp4", ".avi")):
                rows.append({
                    "video_path": os.path.join(manip_dir, vid),
                    "video_id": parse_ffplus_video_id(vid),
                    "fake_label": 1,
                    "manipulation": manip,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No videos found under {ffpp_root}. Expected "
            f"'original_sequences/youtube/{compression}/videos/' and "
            f"'manipulated_sequences/<method>/{compression}/videos/'. "
            "Check --ffpp_root and --compression."
        )
    return df


def convert_ffplus(
    ffpp_root: str,
    output_dir: str,
    manipulation: str = "all",
    compression: str = "c23",
    frames_per_video: int = 10,
    face_size: int = 380,
    sample_n: int | None = None,
    seed: int = 42,
) -> FFPlusStats:
    """Extract face crops from FF++ videos, annotate each with Week 1's ITA
    pipeline, and write canonical labels.csv + annotations.csv.

    face_id format: '<video_id>__f<frame_idx>_<manipulation>' -- the '__'
    separator matches identity_split.py's default face_id_prefix grouping,
    so `python -m src.data.identity_split --group_by face_id_prefix` groups
    all frames (across ALL manipulations) of one source actor together
    automatically, with no extra flags.
    """
    from src.data.frame_extractor import FaceExtractor, extract_face_crops_from_video

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    stats = FFPlusStats()
    df = build_ffplus_metadata(ffpp_root, manipulation, compression)
    stats.total_videos = len(df)

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    extractor = FaceExtractor(image_size=face_size)
    label_rows = []
    annotation_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing FF++ videos"):
        video_path = row["video_path"]
        video_id = row["video_id"]
        fake_label = row["fake_label"]
        manip = row["manipulation"]

        crops = extract_face_crops_from_video(
            video_path, max_frames=frames_per_video, target_size=face_size, extractor=extractor
        )
        stats.frames_extracted += len(crops)

        for frame_idx, face_bgr in enumerate(crops):
            face_id = f"{video_id}__f{frame_idx:03d}_{manip}"
            frame_path = os.path.join(frames_dir, f"{face_id}.png")
            cv2.imwrite(frame_path, face_bgr)

            result = annotate_face(face_bgr, face_id=face_id)

            label_rows.append({
                "face_id": face_id,
                "fake_label": fake_label,
                "video_id": video_id,
                "manipulation": manip,
                "frame_idx": frame_idx,
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

    print(f"[ffplus_adapter] videos: {stats.total_videos}, frames: {stats.frames_extracted}, "
          f"faces OK: {stats.faces_detected}, failed: {stats.annotation_failed}")
    print(f"[ffplus_adapter] wrote {len(labels_df)} rows -> {labels_path}")
    print(f"[ffplus_adapter] wrote {len(annotations_df)} rows -> {annotations_path}")

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ffpp_root", required=True, help="Root of the FaceForensics++ download")
    p.add_argument("--output_dir", required=True, help="data_root to write frames/ + labels.csv/annotations.csv into")
    p.add_argument("--manipulation", default="all", choices=MANIPULATION_TYPES + ["all"])
    p.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    p.add_argument("--frames_per_video", type=int, default=10)
    p.add_argument("--face_size", type=int, default=380)
    p.add_argument("--sample_n", type=int, default=None, help="Subsample N videos for a fast first pass")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_arg_parser().parse_args()
    convert_ffplus(
        ffpp_root=args.ffpp_root,
        output_dir=args.output_dir,
        manipulation=args.manipulation,
        compression=args.compression,
        frames_per_video=args.frames_per_video,
        face_size=args.face_size,
        sample_n=args.sample_n,
        seed=args.seed,
    )


if __name__ == "__main__":
    # Metadata-building smoke test that does NOT require torch/facenet-pytorch
    # or a real FF++ download: builds a synthetic FF++-shaped directory tree
    # (empty video files -- enough to exercise build_ffplus_metadata's
    # traversal and identity-key parsing) and checks the DataFrame it
    # produces. The full convert_ffplus() path additionally needs MTCNN --
    # exercise that separately where torch is installed.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        orig_dir = os.path.join(tmp, "original_sequences", "youtube", "c23", "videos")
        os.makedirs(orig_dir, exist_ok=True)
        for vid_id in ("001", "002", "003"):
            open(os.path.join(orig_dir, f"{vid_id}.mp4"), "w").close()

        for manip in ("Deepfakes", "Face2Face"):
            manip_dir = os.path.join(tmp, "manipulated_sequences", manip, "c23", "videos")
            os.makedirs(manip_dir, exist_ok=True)
            open(os.path.join(manip_dir, "001_002.mp4"), "w").close()
            open(os.path.join(manip_dir, "002_003.mp4"), "w").close()

        df = build_ffplus_metadata(tmp, manipulation="all", compression="c23")
        assert len(df) == 3 + 2 + 2, f"expected 7 rows, got {len(df)}"
        assert set(df[df["fake_label"] == 0]["video_id"]) == {"001", "002", "003"}
        assert set(df[df["fake_label"] == 1]["video_id"]) == {"001", "002"}
        assert parse_ffplus_video_id("001_002.mp4") == "001"
        assert parse_ffplus_video_id("003.mp4") == "003"

        print("[smoke test] build_ffplus_metadata produced:")
        print(df)
        print("[smoke test] PASSED")
