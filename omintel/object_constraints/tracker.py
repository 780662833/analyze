"""Point-tracking adapter interface (module C).

``PointTracker`` tracks seed points across a time range and reports per-point
visibility and confidence. The concrete backend (OpenCV Lucas-Kanade, or later
a denser tracker) is swapped behind this interface; the object-level constraint
logic depends only on the returned ``dict`` shape.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


class PointTracker(Protocol):
    def track(
        self,
        frames: Sequence[Any],
        seed_points: Sequence[Sequence[float]],
        time_range: tuple[int, int],
    ) -> dict[str, Any]:
        """Return ``{"points", "visibility", "confidence"}`` per seed point."""
        ...


class NullPointTracker:
    """Deterministic stub: points stay put, fully visible and confident.

    Output shapes match ``OpenCVLucasKanadeTracker``: ``points`` ``(n_points,
    n_frames, 2)``, ``visibility``/``confidence`` ``(n_points, n_frames)``.
    """

    def track(
        self,
        frames: Sequence[Any],
        seed_points: Sequence[Sequence[float]],
        time_range: tuple[int, int],
    ) -> dict[str, Any]:
        n = len(seed_points)
        n_frames = int(time_range[1]) - int(time_range[0]) + 1
        return {
            "points": [[list(p) for _ in range(n_frames)] for p in seed_points],
            "visibility": [[1.0] * n_frames for _ in range(n)],
            "confidence": [[1.0] * n_frames for _ in range(n)],
        }


def _to_gray(cv2: Any, frame: Any) -> Any:
    """Coerce a frame (numpy array) to single-channel uint8 grayscale."""
    arr = np.asarray(frame)
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


class OpenCVLucasKanadeTracker:
    """Sparse optical-flow tracker (OpenCV Lucas-Kanade).

    ``frames`` is an indexable sequence of numpy frames and ``time_range`` is
    the inclusive ``(start, end)`` slice to track over. ``seed_points`` are
    ``(x, y)`` coordinates in ``frames[start]``. Returns
    ``{"points", "visibility", "confidence"}``, each shaped ``(n_points, n_frames)``.
    """

    def __init__(self, win_size: tuple[int, int] = (15, 15), max_level: int = 3) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError("opencv-python is required for OpenCVLucasKanadeTracker") from exc
        self._cv2 = cv2
        self._win_size = win_size
        self._max_level = max_level

    def track(
        self,
        frames: Sequence[Any],
        seed_points: Sequence[Sequence[float]],
        time_range: tuple[int, int],
    ) -> dict[str, Any]:
        cv2 = self._cv2
        frames = list(frames)
        start, end = int(time_range[0]), int(time_range[1])
        if start < 0 or end >= len(frames) or start > end:
            raise ValueError(f"invalid time_range {time_range} for {len(frames)} frames")

        lk_params = dict(
            winSize=self._win_size,
            maxLevel=self._max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        pts = np.asarray(seed_points, dtype=np.float32).reshape(-1, 1, 2)
        n_points = len(pts)
        all_points: list[list[list[float]]] = [[] for _ in range(n_points)]
        visibility: list[list[float]] = [[] for _ in range(n_points)]
        confidence: list[list[float]] = [[] for _ in range(n_points)]

        gray_prev = _to_gray(cv2, frames[start])
        for i in range(start, end + 1):
            gray = _to_gray(cv2, frames[i])
            if i == start:
                status = np.ones((n_points, 1), dtype=np.uint8)
                err = np.zeros((n_points, 1), dtype=np.float32)
            else:
                pts, status, err = cv2.calcOpticalFlowPyrLK(
                    gray_prev, gray, pts, None, **lk_params
                )
            for k in range(n_points):
                x, y = pts[k].ravel()
                ok = bool(status[k].ravel()[0] == 1)
                all_points[k].append([float(x), float(y)])
                visibility[k].append(1.0 if ok else 0.0)
                confidence[k].append(float(1.0 / (1.0 + err[k].ravel()[0])) if ok else 0.0)
            gray_prev = gray

        return {"points": all_points, "visibility": visibility, "confidence": confidence}