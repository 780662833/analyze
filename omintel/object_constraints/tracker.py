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
        """Return ``{"points", "visibility", "confidence"}`` per seed point.

        Optional contract: backends may also emit a forward-backward
        consistency check as ``fb_error`` ``(n_points, n_frames)`` (see
        ``OpenCVLucasKanadeTracker``); consumers must treat it as optional.
        """
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
            # a stub never drifts: zero round-trip residual, matching the
            # optional fb_error contract of OpenCVLucasKanadeTracker
            "fb_error": [[0.0] * n_frames for _ in range(n)],
            "fb_tolerance_px": 0.0,
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
    ``{"points", "visibility", "confidence"}``, each shaped ``(n_points, n_frames)``,
    plus an optional forward-backward consistency check ``fb_error``
    ``(n_points, n_frames)``: every frame's point is re-tracked back to the
    first frame; a large round-trip residual flags suspected drift so the
    quality layer can surface it instead of trusting a fabricated smooth
    track. Frames where tracking already failed carry ``NaN``.
    """

    def __init__(
        self,
        win_size: tuple[int, int] = (15, 15),
        max_level: int = 3,
        fb_tolerance_px: float = 2.0,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError("opencv-python is required for OpenCVLucasKanadeTracker") from exc
        self._cv2 = cv2
        self._win_size = win_size
        self._max_level = max_level
        self._fb_tolerance_px = float(fb_tolerance_px)

    def _lk(self, g0: Any, g1: Any, pts: Any) -> tuple[Any, Any, Any]:
        cv2 = self._cv2
        return cv2.calcOpticalFlowPyrLK(
            g0, g1, pts, None,
            winSize=self._win_size,
            maxLevel=self._max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def _back_to_start(self, grays: list[Any], pts: Any) -> Any:
        """Re-track ``pts`` (in the last frame) backwards to frame 0."""
        cur = pts.copy()
        for idx in range(len(grays) - 1, 0, -1):
            cur, _, _ = self._lk(grays[idx], grays[idx - 1], cur)
        return cur

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

        pts = np.asarray(seed_points, dtype=np.float32).reshape(-1, 1, 2)
        n_points = len(pts)
        all_points: list[list[list[float]]] = [[] for _ in range(n_points)]
        visibility: list[list[float]] = [[] for _ in range(n_points)]
        confidence: list[list[float]] = [[] for _ in range(n_points)]

        grays = [_to_gray(cv2, f) for f in frames[start:end + 1]]
        cur = pts.copy()
        gray_prev = grays[0]
        per_frame_pts: list[Any] = []          # tracked points at each frame
        per_frame_status: list[Any] = []       # LK status at each frame
        for idx, gray in enumerate(grays):
            if idx == 0:
                status = np.ones((n_points, 1), dtype=np.uint8)
                err = np.zeros((n_points, 1), dtype=np.float32)
            else:
                cur, status, err = self._lk(gray_prev, gray, cur)
            per_frame_pts.append(cur.copy())
            per_frame_status.append(status.copy())
            for k in range(n_points):
                x, y = cur[k].ravel()
                ok = bool(status[k].ravel()[0] == 1)
                all_points[k].append([float(x), float(y)])
                visibility[k].append(1.0 if ok else 0.0)
                confidence[k].append(float(1.0 / (1.0 + err[k].ravel()[0])) if ok else 0.0)
            gray_prev = gray

        # Forward-backward consistency: re-track each frame's point back to
        # the first frame; the round-trip residual flags suspected drift.
        origin = np.asarray(seed_points, dtype=np.float32).reshape(-1, 2)
        fb_error: list[list[float]] = [[] for _ in range(n_points)]
        for f, (pf_pts, pf_status) in enumerate(zip(per_frame_pts, per_frame_status)):
            back = self._back_to_start(grays[: f + 1], pf_pts)
            residuals = np.linalg.norm(back.reshape(-1, 2) - origin, axis=1)
            for k in range(n_points):
                tracked_ok = bool(pf_status[k].ravel()[0] == 1)
                fb_error[k].append(float(residuals[k]) if tracked_ok else float("nan"))

        return {
            "points": all_points,
            "visibility": visibility,
            "confidence": confidence,
            "fb_error": fb_error,
            "fb_tolerance_px": self._fb_tolerance_px,
        }