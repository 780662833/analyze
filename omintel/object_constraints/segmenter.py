"""Video object-segmentation adapter interface (module B).

``Segmenter`` turns a seed (a detection box or a manual prompt) into per-frame
masks. The concrete backend (SAM 2 video predictor, or a stub) lives behind
this interface; the returned mask type is backend-specific and consumed only
by later adapters, never serialized into the JSON contracts.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .detector import BBox


class Segmenter(Protocol):
    def seed(self, frame: Any, bbox: BBox) -> Any:
        """Create a video-propagation state from an initial box on a frame."""
        ...

    def propagate(self, state: Any, frame: Any) -> Any:
        """Return the mask for ``frame``, advancing internal state."""
        ...


class StaticBoxSegmenter:
    """Deterministic stub: the mask is the seeded box, unchanged per frame."""

    def seed(self, frame: Any, bbox: BBox) -> Any:
        return bbox

    def propagate(self, state: Any, frame: Any) -> Any:
        return state


class BoxMaskSegmenter:
    """Deterministic stub that returns a real binary mask for the seeded box.

    Unlike ``StaticBoxSegmenter`` (which returns the box itself), this produces
    an actual per-frame ``(H, W)`` ``uint8`` mask (255 inside the box), suitable
    for exercising the mask-to-Disk write path in a model-free closed loop.
    """

    def seed(self, frame: Any, bbox: BBox) -> Any:
        return bbox

    def propagate(self, state: Any, frame: Any) -> Any:
        h, w = np.asarray(frame).shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x0 = int(round(state.x_min * w))
        y0 = int(round(state.y_min * h))
        x1 = int(round(state.x_max * w))
        y1 = int(round(state.y_max * h))
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        mask[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = 255
        return mask