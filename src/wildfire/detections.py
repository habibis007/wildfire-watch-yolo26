"""A tiny, dependency-free detection type plus an adapter for Roboflow results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Detection:
    """One bounding box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def scaled(self, sx: float, sy: float) -> "Detection":
        """Return the box rescaled, e.g. from the resized inference frame back to the original."""
        return Detection(
            self.class_name,
            self.confidence,
            self.x1 * sx,
            self.y1 * sy,
            self.x2 * sx,
            self.y2 * sy,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Detection":
        return cls(**d)


def _get(obj: Any, key: str) -> Any:
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def from_inference_result(result: Any, min_confidence: float = 0.0) -> list[Detection]:
    """Convert a Roboflow ``inference`` result into ``Detection`` objects.

    Works with both the pydantic response object returned by
    ``model.infer(frame)[0]`` and its plain-dict form. Roboflow reports boxes as
    centre x/y plus width/height, so we convert to corner coordinates here.
    """
    predictions: Iterable[Any] = _get(result, "predictions")
    out: list[Detection] = []
    for p in predictions:
        conf = float(_get(p, "confidence"))
        if conf < min_confidence:
            continue
        cx, cy = float(_get(p, "x")), float(_get(p, "y"))
        w, h = float(_get(p, "width")), float(_get(p, "height"))
        # The pydantic object exposes ``class_name``; its JSON/dict form uses ``class``.
        name = p.get("class", p.get("class_name")) if isinstance(p, dict) else _get(p, "class_name")
        out.append(
            Detection(
                class_name=str(name),
                confidence=conf,
                x1=cx - w / 2,
                y1=cy - h / 2,
                x2=cx + w / 2,
                y2=cy + h / 2,
            )
        )
    return out
