"""Object-detection adapter interface (module A).

``Detector`` locates candidate objects as normalized ``xyxy`` boxes. The
concrete backend (Grounding DINO, an existing detector, or a manual box) is
swapped behind this interface so the rest of the pipeline never depends on a
specific model.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Protocol, Sequence


class BBox(NamedTuple):
    """Axis-aligned box, normalized to ``[0, 1]`` against coded resolution."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detector(Protocol):
    def detect(self, frame: Any, prompt_texts: Sequence[str]) -> list[BBox]:
        """Return candidate boxes for the given text prompts on one frame."""
        ...


class StaticBoxDetector:
    """Deterministic stub for wiring tests: always returns one fixed box."""

    def __init__(self, bbox: BBox) -> None:
        self._bbox = bbox

    def detect(self, frame: Any, prompt_texts: Sequence[str]) -> list[BBox]:
        return [self._bbox]