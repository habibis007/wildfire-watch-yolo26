"""Alert logic: only raise an alert when fire or smoke persists across sampled frames.

A single-frame detection is often a false positive (sun glare, a red roof, low
cloud). Requiring the same class in ``min_consecutive`` sampled frames in a row
trades a little latency for far fewer false alarms. An alert stays active until
the class has been absent for ``clear_after`` sampled frames, so a box that
flickers for one frame does not end the event and immediately start a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .detections import Detection


@dataclass
class AlertEvent:
    class_name: str
    start_time: float  # timestamp of the FIRST frame in the qualifying streak
    raised_time: float  # timestamp when the alert actually fired
    end_time: float | None = None  # timestamp of the last frame with a detection
    peak_confidence: float = 0.0
    frames_with_detection: int = 0

    def describe(self) -> str:
        end = f"{self.end_time:6.2f}s" if self.end_time is not None else "  open "
        return (
            f"[ALERT] {self.class_name.upper():5s} first seen {self.start_time:6.2f}s, "
            f"alert raised {self.raised_time:6.2f}s, last seen {end}, "
            f"peak conf {self.peak_confidence:.2f}"
        )


@dataclass
class _Track:
    streak: int = 0
    misses: int = 0
    streak_start: float = 0.0
    streak_peak: float = 0.0
    active: AlertEvent | None = None


@dataclass
class PersistenceAlert:
    classes: tuple[str, ...] = ("fire", "smoke")
    min_consecutive: int = 5
    clear_after: int = 3
    min_confidence: float = 0.4
    events: list[AlertEvent] = field(default_factory=list)
    _tracks: dict[str, _Track] = field(default_factory=dict, init=False, repr=False)
    _last_ts: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_consecutive < 1 or self.clear_after < 1:
            raise ValueError("min_consecutive and clear_after must be >= 1")
        self._tracks = {c: _Track() for c in self.classes}

    def update(self, timestamp: float, detections: Iterable[Detection]) -> list[AlertEvent]:
        """Feed one SAMPLED frame. Returns events newly raised on this frame."""
        self._last_ts = timestamp
        best: dict[str, float] = {}
        for d in detections:
            if d.class_name in self._tracks and d.confidence >= self.min_confidence:
                best[d.class_name] = max(best.get(d.class_name, 0.0), d.confidence)

        raised: list[AlertEvent] = []
        for cls, tr in self._tracks.items():
            conf = best.get(cls)
            if conf is not None:
                tr.misses = 0
                if tr.streak == 0:
                    tr.streak_start, tr.streak_peak = timestamp, 0.0
                tr.streak += 1
                tr.streak_peak = max(tr.streak_peak, conf)
                if tr.active is not None:
                    tr.active.end_time = timestamp
                    tr.active.peak_confidence = max(tr.active.peak_confidence, conf)
                    tr.active.frames_with_detection += 1
                elif tr.streak >= self.min_consecutive:
                    ev = AlertEvent(
                        class_name=cls,
                        start_time=tr.streak_start,
                        raised_time=timestamp,
                        end_time=timestamp,
                        peak_confidence=tr.streak_peak,
                        frames_with_detection=tr.streak,
                    )
                    tr.active = ev
                    self.events.append(ev)
                    raised.append(ev)
            else:
                tr.streak = 0
                if tr.active is not None:
                    tr.misses += 1
                    if tr.misses >= self.clear_after:
                        tr.active = None
                        tr.misses = 0
        return raised

    @property
    def active_classes(self) -> list[str]:
        return [c for c, tr in self._tracks.items() if tr.active is not None]

    def status_text(self) -> str:
        active = self.active_classes
        if active:
            return "ALERT: " + " + ".join(c.upper() for c in active) + f" detected  t={self._last_ts:.1f}s"
        return f"Monitoring...  t={self._last_ts:.1f}s"

    def summary(self) -> str:
        if not self.events:
            return "No persistent fire or smoke detected."
        return "\n".join(e.describe() for e in self.events)
