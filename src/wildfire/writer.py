"""Safe video output.

The old repo's output videos were truncated because ``cv2.VideoWriter`` was
never released: the container's index is only written on ``release()``. Here
the writer is a context manager, so ``release()`` runs even if the loop raises
or the notebook cell is interrupted.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class AnnotatedVideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int] | None = None, fourcc: str = "mp4v"):
        self.path = Path(path)
        self.fps = fps
        self.size = size  # (width, height); taken from the first frame if None
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self.frames_written = 0

    def _open(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self.size = self.size or (w, h)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, self.size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.path} (codec {self.fourcc})")

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._open(frame)
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self._writer.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def to_h264(src: str | Path, dst: str | Path | None = None, crf: int = 23) -> Path:
    """Re-encode to H.264 + yuv420p so it plays in browsers, Colab and GitHub.

    OpenCV's ``mp4v`` output does not play in most browsers. Returns the source
    path unchanged if ffmpeg is not installed.
    """
    src = Path(src)
    if shutil.which("ffmpeg") is None:
        return src
    dst = Path(dst) if dst else src.with_name(src.stem + "_h264.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", str(dst)],
        check=True,
    )
    return dst


def to_gif(src: str | Path, dst: str | Path, width: int = 640, fps: int = 10, start: float = 0, duration: float = 8) -> Path:
    """Make a small looping GIF (for the README) using ffmpeg's palette filter."""
    dst = Path(dst)
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(duration), "-i", str(src),
         "-vf", vf, "-loop", "0", str(dst)],
        check=True,
    )
    return dst
