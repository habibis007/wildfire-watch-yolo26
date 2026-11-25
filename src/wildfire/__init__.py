"""Wildfire and smoke detection helpers for the drone-video demo.

The model itself is loaded from Roboflow by its ID (see the notebook); this
package only holds the model-agnostic pieces: frame sampling, preprocessing,
drawing, persistence-based alerting and safe video writing.
"""

from .alerts import AlertEvent, PersistenceAlert
from .detections import Detection, from_inference_result
from .draw import CLASS_COLORS, draw_alert_banner, draw_detections
from .frames import SampledFrame, sample_frames, video_info
from .preprocess import apply_clahe, resize_keep_aspect
from .writer import AnnotatedVideoWriter, to_h264

__all__ = [
    "AlertEvent",
    "AnnotatedVideoWriter",
    "CLASS_COLORS",
    "Detection",
    "PersistenceAlert",
    "SampledFrame",
    "apply_clahe",
    "draw_alert_banner",
    "draw_detections",
    "from_inference_result",
    "resize_keep_aspect",
    "sample_frames",
    "to_h264",
    "video_info",
]
