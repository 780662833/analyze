"""L1.5 Commit E: deterministic frame-measurement sequences (E1 core).

This module computes per-frame luminance statistics and adjacent-frame
brightness deltas over the *authoritative L0 PTS timeline*.  It produces a
measurement record only; event-candidate derivation lives in
``event_extractors`` so that reviewers can audit the raw sequence before any
threshold is applied to it.

Contract rules honored here (see event_models.py):
  * Time base is L0 ``pts_time_s`` consumed verbatim from frames.csv rows --
    never re-derived as frame_no / fps.
  * Frames whose PTS is missing/unparseable are NOT silently dropped; they
    become explicit CoverageGap entries with reason ``missing_pts``.
  * All outputs are deterministic for identical inputs, params, and
    ALGORITHM_VERSION (floats are rounded at one canonical point, lists keep
    frame order, dicts serialize via event_models._jsonable).
  * No LLM, no model downloads, no GPU.  Pure numpy over decoded grayscale.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

#: bump whenever measurement semantics change; recorded in provenance.
ALGORITHM_VERSION = "1.0.0"

DEFAULT_PARAMS: dict[str, Any] = {
    # round stats to this many decimals so float noise cannot leak into output
    "stat_precision": 6,
}


def _rounded(value: float, precision: int) -> float:
    return float(round(float(value), precision))


def compute_luminance_sequence(
    frames: list[dict[str, Any]],
    images: list[np.ndarray | None],
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute mean/std luminance per frame and adjacent deltas.

    Parameters
    ----------
    frames:
        L0 frames.csv rows (dicts with at least ``frame_no`` and
        ``pts_time_s``; extra columns are ignored).  Order defines adjacency.
    images:
        One entry per frame row: grayscale uint8 ndarray, or ``None`` when the
        frame could not be read/decoded.  Length must equal ``len(frames)``.
    params:
        Overrides merged over DEFAULT_PARAMS.

    Returns
    -------
    dict
        Measurement record with keys:
          algorithm, algorithm_version, params, frame_nos, pts_s,
          mean_luma, std_luma, delta_mean_luma, invalid_frames, gaps,
          coverage_ratio, input_digest.
        ``pts_s`` entries are None exactly where L0 PTS was missing; the
        corresponding frame ranges appear in ``gaps`` (reason=missing_pts).
        ``delta_mean_luma[i]`` = mean_luma[i+1] - mean_luma[i]; deltas that
        touch an unreadable frame are None and gap-logged as
        measurement_unavailable.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    precision = int(p["stat_precision"])

    if len(frames) != len(images):
        raise ValueError(
            f"frames/images length mismatch: {len(frames)} vs {len(images)}"
        )

    frame_nos: list[int] = []
    pts_s: list[float | None] = []
    mean_luma: list[float | None] = []
    std_luma: list[float | None] = []
    invalid_frames: list[int] = []
    gaps: list[dict[str, Any]] = []

    def _gap(reason: str, fn: int, detail: str) -> None:
        gaps.append({
            "reason": reason,
            "frame_range": {"start": fn, "end": fn + 1, "domain": "frame_no"},
            "detail": detail,
        })

    for row, img in zip(frames, images):
        fn = int(row["frame_no"])
        frame_nos.append(fn)

        raw_pts = row.get("pts_time_s")
        try:
            pts = float(raw_pts)
            if not np.isfinite(pts):
                pts = None
        except (TypeError, ValueError):
            pts = None
        pts_s.append(pts)
        if pts is None:
            _gap("missing_pts", fn, "L0 frames.csv row has no usable pts_time_s")

        if img is None:
            mean_luma.append(None)
            std_luma.append(None)
            invalid_frames.append(fn)
            _gap("measurement_unavailable", fn, "frame image unreadable/undecoded")
            continue

        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"frame {fn}: expected 2-D grayscale, got shape {arr.shape}")
        mean_luma.append(_rounded(arr.mean(), precision))
        std_luma.append(_rounded(arr.std(), precision))

    delta: list[float | None] = []
    for i in range(len(frame_nos) - 1):
        a, b = mean_luma[i], mean_luma[i + 1]
        if a is None or b is None:
            delta.append(None)
            _gap(
                "measurement_unavailable",
                frame_nos[i + 1],
                "adjacent-pair delta skipped due to unreadable neighbor "
                f"(frames {frame_nos[i]}->{frame_nos[i + 1]})",
            )
        else:
            delta.append(_rounded(b - a, precision))

    n_pairs = max(len(delta), 1)
    valid_pairs = sum(1 for d in delta if d is not None)

    digest = hashlib.sha256(
        repr((frame_nos, pts_s, mean_luma, std_luma)).encode("utf-8")
    ).hexdigest()

    return {
        "algorithm": "luminance_sequence",
        "algorithm_version": ALGORITHM_VERSION,
        "params": p,
        "frame_nos": frame_nos,
        "pts_s": pts_s,
        "mean_luma": mean_luma,
        "std_luma": std_luma,
        "delta_mean_luma": delta,
        "invalid_frames": invalid_frames,
        "gaps": gaps,
        "coverage_ratio": _rounded(valid_pairs / n_pairs, 6) if delta else 0.0,
        "input_digest": digest,
    }


def check_measurement_determinism(record: dict[str, Any]) -> None:
    """Re-serialize check helper used by tests: digest must match content."""
    expect = hashlib.sha256(
        repr((
            record["frame_nos"], record["pts_s"],
            record["mean_luma"], record["std_luma"],
        )).encode("utf-8")
    ).hexdigest()
    if record["input_digest"] != expect:
        raise AssertionError("measurement record is not self-consistent")
