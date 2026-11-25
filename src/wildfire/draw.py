"""Visualisation with plain OpenCV: colour-coded boxes, labels and an alert banner."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .detections import Detection

# BGR colours. Fire is red-orange, smoke is a light slate grey-blue so it stays
# visible against both grey smoke and green/brown terrain.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "fire": (0, 69, 255),
    "smoke": (200, 180, 150),
}
DEFAULT_COLOR = (0, 255, 255)


def draw_detections(image: np.ndarray, detections: Iterable[Detection], thickness: int | None = None) -> np.ndarray:
    """Return a copy of ``image`` with boxes and "class 0.87" labels drawn on it."""
    out = image.copy()
    h, w = out.shape[:2]
    t = thickness or max(2, round(min(h, w) / 300))
    font_scale = max(0.5, min(h, w) / 1000)
    for d in detections:
        color = CLASS_COLORS.get(d.class_name, DEFAULT_COLOR)
        p1 = (int(max(0, d.x1)), int(max(0, d.y1)))
        p2 = (int(min(w - 1, d.x2)), int(min(h - 1, d.y2)))
        cv2.rectangle(out, p1, p2, color, t)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = p1[1] - 4 if p1[1] - th - base - 4 > 0 else p1[1] + th + base + 4
        cv2.rectangle(out, (p1[0], ty - th - base), (p1[0] + tw + 6, ty + base), color, -1)
        cv2.putText(out, label, (p1[0] + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_alert_banner(image: np.ndarray, text: str, active: bool) -> np.ndarray:
    """Draw a status bar across the top: red when an alert is active, dark otherwise."""
    out = image.copy()
    h, w = out.shape[:2]
    bar_h = max(28, h // 18)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 200) if active else (40, 40, 40), -1)
    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    font_scale = bar_h / 45
    cv2.putText(out, text, (10, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out
