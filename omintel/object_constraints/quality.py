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


def assess_window(
    frame_nos: Sequence[int],
    pts_ts: Sequence[float],
    masks: Sequence[Any],
    motion: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Compute one ``quality_per_frame.csv`` row per frame.

    ``mask_area_ratio_change`` is relative to the first frame's mask area; a
    per-frame ``flow_agreement`` is derived from the tracker's confidence when
    available. Appearance loss stays ``None`` (no appearance model in v1).
    """
    n_frames = len(frame_nos)
    if len(pts_ts) != n_frames or len(masks) != n_frames:
        raise ValueError("frame_nos/pts_ts/masks lengths must match")

    areas = [
        float(np.asarray(m).astype(bool).sum()) / max(1, int(np.asarray(m).size))
        for m in masks
    ]
    base = areas[0] if areas else 0.0

    conf = (motion or {}).get("confidence") or []
    n_points = len(conf)

    rows: list[dict[str, Any]] = []
    for f in range(n_frames):
        flow = (
            float(np.mean([conf[p][f] for p in range(n_points)]))
            if n_points
            else None
        )
        area_change = (areas[f] - base) / base if base else 0.0
        rows.append(
            assess_frame(
                frame_no=frame_nos[f],
                pts_time_s=pts_ts[f],
                mask_area_ratio_change=round(area_change, 6),
                flow_agreement=round(flow, 6) if flow is not None else None,
            )
        )
    return rows