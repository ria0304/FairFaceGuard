"""Video -> face-crop extraction for video-based deepfake datasets.

AI-Face ships pre-cropped still images, so nothing in Weeks 1-5 ever had to
open a video file. FaceForensics++ and Celeb-DF ship raw .mp4 videos, so
switching to them adds exactly one new step in front of everything else:
sample N frames per video, run face detection on each, and write out
square face crops in the same `frames/<face_id>.png` layout
`src/data/datasets.py` already expects. Nothing downstream of this file
needs to know the source was ever a video.

Uses facenet-pytorch's MTCNN for face detection, since it's the detector
most commonly used to prepare FF++/Celeb-DF for deepfake-detection papers
and keeps face-crop conventions consistent with the published benchmarks.
This is a hard dependency for this module only (`pip install facenet-pytorch`) --
the rest of the pipeline does not need it.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


class FaceExtractor:
    """Thin wrapper around MTCNN so callers don't need to know about
    torch tensor <-> BGR-numpy conversion."""

    def __init__(self, image_size: int = 380, margin: int = 0, device: str | None = None):
        try:
            import torch
            from facenet_pytorch import MTCNN
        except ImportError as e:
            raise ImportError(
                "frame_extractor requires 'torch' and 'facenet-pytorch'. "
                "Install with: pip install torch facenet-pytorch"
            ) from e

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, post_process=False, device=self.device)
        self.image_size = image_size

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Returns a single face crop (BGR, uint8, image_size x image_size)
        or None if no face was detected. If multiple faces are present,
        MTCNN's default `select_largest=True` behavior returns the most
        prominent one -- fine for FF++/Celeb-DF, which are single-subject
        clips."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face = self.mtcnn(frame_rgb, save_path=None)
        if face is None:
            return None
        face_np = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        return face_bgr


def extract_face_crops_from_video(
    video_path: str,
    max_frames: int = 10,
    target_size: int = 380,
    extractor: FaceExtractor | None = None,
) -> list[np.ndarray]:
    """Sample `max_frames` evenly-spaced frames from a video and return the
    face crop from each frame that had a detectable face. Frames with no
    detected face are silently dropped (not padded), so the returned list
    can be shorter than max_frames -- callers should not assume a fixed
    count per video.

    Reuses a single MTCNN instance across the whole video (`extractor`) when
    processing many videos in a loop, since constructing MTCNN per-frame is
    the dominant cost otherwise.
    """
    if extractor is None:
        extractor = FaceExtractor(image_size=target_size)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    crops: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        face = extractor.extract(frame)
        if face is not None:
            crops.append(face)
    cap.release()
    return crops


if __name__ == "__main__":
    # Smoke test that does NOT require torch/facenet-pytorch: builds a
    # synthetic .mp4 and checks the frame-sampling / video-reading path in
    # isolation via a stub extractor. The MTCNN-dependent path
    # (FaceExtractor itself) can only be exercised where torch is installed.
    import tempfile

    class _StubExtractor:
        """Mimics FaceExtractor's interface without needing MTCNN."""

        def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
            return frame_bgr  # pretend every frame has a detectable face

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "synthetic.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 10, (64, 64))
        for i in range(30):
            frame = np.full((64, 64, 3), i % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        crops = extract_face_crops_from_video(
            video_path, max_frames=5, target_size=64, extractor=_StubExtractor()
        )
        assert len(crops) == 5, f"expected 5 crops, got {len(crops)}"
        assert all(c.shape == (64, 64, 3) for c in crops)
        print(f"[smoke test] extracted {len(crops)} frames from synthetic video")
        print("[smoke test] PASSED")
