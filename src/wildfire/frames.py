"""Frame extraction: read a video and yield every Nth frame with its timestamp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class SampledFrame:
    index: int  # frame number in the source video
    timestamp: float  # seconds from the start of the video
    image: np.ndarray  # BGR, as OpenCV reads it
    is_sampled: bool  # True if this frame should be sent to the model


def video_info(path: str | Path) -> dict:
    """Return fps, frame count, width, height and duration of a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "fps": fps,
            "frame_count": frames,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": frames / fps if fps else 0.0,
        }
    finally:
        cap.release()


def sample_frames(
    path: str | Path,
    every_n: int = 5,
    include_unsampled: bool = False,
    max_seconds: float | None = None,
) -> Iterator[SampledFrame]:
    """Yield frames from ``path``.

    Every ``every_n``-th frame is marked ``is_sampled=True`` (the ones we run the
    model on). With ``include_unsampled=True`` the frames in between are yielded
    too, so the output video can keep the original frame rate while reusing the
    most recent boxes. The capture is always released, even if the caller stops
    iterating early.
    """
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = index / fps
            if max_seconds is not None and ts > max_seconds:
                break
            sampled = index % every_n == 0
            if sampled or include_unsampled:
                yield SampledFrame(index=index, timestamp=ts, image=frame, is_sampled=sampled)
            index += 1
    finally:
        cap.release()
