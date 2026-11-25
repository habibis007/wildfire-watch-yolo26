"""Preprocessing: resizing and CLAHE contrast enhancement for hazy smoke."""

from __future__ import annotations

import cv2
import numpy as np


def resize_keep_aspect(image: np.ndarray, max_side: int = 640) -> tuple[np.ndarray, float]:
    """Shrink ``image`` so its longest side is ``max_side``; returns (image, scale).

    ``scale`` is new/original, so divide box coordinates by it to map back.
    Images already small enough are returned unchanged with scale 1.0.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Boost local contrast on the lightness channel only (LAB), keeping colours natural.

    Haze flattens contrast; CLAHE (Contrast Limited Adaptive Histogram
    Equalisation) brings out the edges of thin smoke plumes without blowing out
    the sky the way global histogram equalisation would.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
