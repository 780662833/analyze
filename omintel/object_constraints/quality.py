"""Per-frame quality assessment and gate aggregation (module E).

The per-frame scalar dimensions are computed here; the anomaly score and the
PASS/REPAIR/REVIEW gate stay *uncalibrated* until a labeled test set supplies
thresholds. ``assess_frame`` therefore emits ``anomaly_score: null`` and
``gate: "REVIEW"`` by default: no objective number is fabricated, matching the
contract's stance in ``schemas.py``.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .schemas import UNcalibrated_REASON


def assess_frame(
    *,
    frame_no: int,
    pts_time_s: float,
    mask_area_ratio_change: float | None = None,
    flow_agreement: float | None = None,
    feature_loss_ratio: float | None = None,
    gate: str = "REVIEW",
    gate_reason: str = UNcalibrated_REASON,
) -> dict[str, Any]:
    """Return one row matching ``quality_per_frame.csv`` columns.

    Weights are uncalibrated, so ``anomaly_score`` is ``None`` and ``gate`` is
    ``REVIEW`` by default; the raw per-dimension measurements are filled by
    callers, and ``gate_reason`` records *why* the gate is what it is.
    """
    return {
        "frame_no": frame_no,
        "pts_time_s": pts_time_s,
        "mask_area_ratio_change": mask_area_ratio_change,
        "flow_agreement": flow_agreement,
        "feature_loss_ratio": feature_loss_ratio,
        "anomaly_score": None,
        "gate": gate,
        "gate_reason": gate_reason,
    }


def aggregate_gate(rows: Sequence[dict[str, Any]]) -> str:
    """Aggregate per-frame gates: REVIEW > REPAIR > PASS."""
    gates = {r.get("gate") for r in rows}
    if "REVIEW" in gates:
        return "REVIEW"
    if "REPAIR" in gates:
        return "REPAIR"
    return "PASS" if gates else "REVIEW"


def _point_scalar(motion: dict[str, Any] | None, key: str, f: int) -> float | None:
    """Mean of ``motion[key]`` over points at frame ``f``; NaN entries skipped.

    Returns ``None`` when the array is absent or every point failed to track
    on that frame (NaN), so "no measurement" stays distinguishable from 0.
    """
    arr = (motion or {}).get(key) or []
    vals = [float(p[f]) for p in arr if f < len(p) and not np.isnan(float(p[f]))]
    return float(np.mean(vals)) if vals else None


def assess_window(
    frame_nos: Sequence[int],
    pts_ts: Sequence[float],
    masks: Sequence[Any],
    motion: dict[str, Any] | None,
    bboxes: Sequence[Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    """Compute one ``quality_per_frame.csv`` row per frame.

    ``mask_area_ratio_change`` is relative to the first frame's mask area; a
    per-frame ``flow_agreement`` is derived from the tracker's confidence when
    available, plus the optional forward-backward consistency signal
    (``fb_error`` mean / ``fb_tolerance_px``) recorded in ``gate_reason`` so
    suspected drift surfaces as REVIEW instead of a fabricated smooth track.
    ``feature_loss_ratio`` is the fraction of seed points that failed to track
    on the frame (visibility < 0.5 or NaN fb residual). Appearance loss stays
    ``None`` (no appearance model in v1). When per-frame ``bboxes`` are given,
    sudden box jumps/size changes surface as additional REVIEW reasons -- these
    rule-based signals stay uncalibrated until a labeled sample sets thresholds.
    """
    n_frames = len(frame_nos)
    if len(pts_ts) != n_frames or len(masks) != n_frames:
        raise ValueError("frame_nos/pts_ts/masks lengths must match")
    if bboxes is not None and len(bboxes) != n_frames:
        raise ValueError("bboxes length must match frame_nos")

    areas = [
        float(np.asarray(m).astype(bool).sum()) / max(1, int(np.asarray(m).size))
        for m in masks
    ]
    base = areas[0] if areas else 0.0

    visibility = (motion or {}).get("visibility") or []
    fb = (motion or {}).get("fb_error") or []
    fb_tol = (motion or {}).get("fb_tolerance_px")

    def _box_metrics(f: int) -> tuple[float | None, float | None]:
        """(center jump in normalized units, area change vs previous box)."""
        if bboxes is None or f == 0:
            return None, None
        prev, cur = bboxes[f - 1], bboxes[f]
        pcx, pcy = (prev[0] + prev[2]) / 2.0, (prev[1] + prev[3]) / 2.0
        ccx, ccy = (cur[0] + cur[2]) / 2.0, (cur[1] + cur[3]) / 2.0
        jump = float(np.hypot(ccx - pcx, ccy - pcy))
        pa = max(0.0, prev[2] - prev[0]) * max(0.0, prev[3] - prev[1])
        ca = max(0.0, cur[2] - cur[0]) * max(0.0, cur[3] - cur[1])
        area_chg = (ca - pa) / pa if pa > 0 else None
        return jump, area_chg

    rows: list[dict[str, Any]] = []
    for f in range(n_frames):
        flow = _point_scalar(motion, "confidence", f)
        fb_mean = _point_scalar(motion, "fb_error", f)
        # feature loss: share of seed points not reliably tracked on this frame
        tracked_any = bool(visibility) or bool(fb)
        if tracked_any:
            n_pts = max(len(visibility), len(fb))
            lost_k = 0
            for k in range(n_pts):
                vis_ok = f < len(visibility[k]) and float(visibility[k][f]) >= 0.5
                fb_row = fb[k] if k < len(fb) else []
                fb_ok = f < len(fb_row) and not np.isnan(float(fb_row[f]))
                if not (vis_ok and fb_ok):
                    lost_k += 1
            feature_loss = lost_k / n_pts if n_pts else None
        else:
            feature_loss = None
        gate, reason = "REVIEW", UNcalibrated_REASON
        lost = bool(visibility) and all(
            p[f] < 0.5 for p in visibility if f < len(p)
        )
        if lost:
            reason = "tracking lost on this frame; re-initialization required; " + reason
        elif fb_mean is not None and fb_tol is not None and fb_mean > float(fb_tol):
            reason = f"forward-backward error {fb_mean:.2f}px exceeds tolerance; suspected drift; " + reason
        jump, area_chg = _box_metrics(f)
        if jump is not None and jump > 0.05:
            reason = f"bbox center jumped {jump:.3f} normalized units; suspected jump; " + reason
        if area_chg is not None and abs(area_chg) > 0.5:
            reason = f"bbox area changed {area_chg:+.2f}; suspected scale jump; " + reason
        rows.append(
            assess_frame(
                frame_no=frame_nos[f],
                pts_time_s=pts_ts[f],
                mask_area_ratio_change=round(area_change := (areas[f] - base) / base if base else 0.0, 6),
                flow_agreement=round(flow, 6) if flow is not None else None,
                feature_loss_ratio=round(feature_loss, 6) if feature_loss is not None else None,
                gate=gate,
                gate_reason=reason,
            )
        )
    return rows