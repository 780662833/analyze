"""Loaders that read the existing L0 artifacts and fill the object contracts.

The object layer does not re-derive media facts. It consumes the canonical
``media_facts.json`` (resolution, rotation, sha256, VFR flag) and ``frames.csv``
(PTS timeline) that ``omintel.media_facts`` already produces. An optional
``provisional_shots.csv`` supplies shot boundaries so the edit window can be
selected by ``shot_id`` instead of raw timestamps.

Time stays PTS-authoritative end to end: window timestamps come from the shot
table or explicit arguments, and each is snapped to the nearest decoded frame
via ``frames.csv``; ``frame_no`` is derived only as a lookup key.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from omintel.fusion import nearest_frame_index

from .schemas import build_object_constraints, build_object_tracks


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def read_frames(frames_csv: str | Path) -> list[dict[str, Any]]:
    """Return one dict per decoded frame: ``frame_no`` (int) and ``pts_time_s`` (float or None)."""
    frames: list[dict[str, Any]] = []
    with Path(frames_csv).open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("pts_time_s")
            pts = None if raw in (None, "") else float(raw)
            frames.append({"frame_no": int(row["frame_no"]), "pts_time_s": pts})
    return frames


def load_l0(media_facts_path: str | Path) -> dict[str, Any]:
    """Distill the ``media_facts.json`` fields the builders need."""
    facts = read_json(media_facts_path)
    source = facts.get("source", {}) or {}
    vs = facts.get("video_stream", {}) or {}
    tl = facts.get("frame_timeline", {}) or {}

    sha256 = source.get("sha256") or ""
    coded_w = int(vs.get("width") or 0)
    coded_h = int(vs.get("height") or 0)
    display_w = int(vs.get("display_width") or coded_w)
    display_h = int(vs.get("display_height") or coded_h)
    rotation = int(vs.get("rotation_deg") or 0)

    return {
        "source_sha256": sha256,
        "video_id": sha256[:16] if sha256 else None,
        "coded_size": (coded_w, coded_h),
        "display_size": (display_w, display_h),
        "rotation_deg": rotation,
        "likely_variable_frame_rate": bool(tl.get("likely_variable_frame_rate")),
    }


def _snap(frames: list[dict[str, Any]], t_s: float) -> dict[str, Any]:
    """Map a PTS timestamp to ``{"pts_s", "frame_no"}`` of the nearest decoded frame."""
    frame_no = nearest_frame_index(float(t_s), frames)
    by_no = {r["frame_no"]: r for r in frames}
    frame = by_no.get(frame_no) if frame_no is not None else None
    if frame is None or frame.get("pts_time_s") is None:
        raise ValueError(f"no frame with a usable PTS near t={t_s}")
    return {"pts_s": float(frame["pts_time_s"]), "frame_no": int(frame["frame_no"])}


def _read_shot_bounds(shots_csv: str | Path, shot_id: str) -> tuple[float, float]:
    with Path(shots_csv).open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("shot_id") == shot_id:
                return float(row["start_s"]), float(row["end_s"])
    raise ValueError(f"shot_id {shot_id!r} not found in {shots_csv}")


def resolve_window(
    frames: list[dict[str, Any]],
    *,
    shot_id: str | None = None,
    shots_csv: str | Path | None = None,
    pts_start: float | None = None,
    pts_end: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve an edit window to ``(start, end)``, each snapped to a decoded frame.

    Precedence: if ``shot_id`` is given it is looked up in ``shots_csv``;
    otherwise ``pts_start``/``pts_end`` are used directly.
    """
    if shot_id is not None:
        if shots_csv is None:
            raise ValueError("shots_csv is required when shot_id is provided")
        start_s, end_s = _read_shot_bounds(shots_csv, shot_id)
    elif pts_start is not None and pts_end is not None:
        start_s, end_s = float(pts_start), float(pts_end)
    else:
        raise ValueError("provide shot_id+shots_csv, or pts_start+pts_end")

    if start_s > end_s:
        raise ValueError(f"window start ({start_s}) exceeds end ({end_s})")
    return _snap(frames, start_s), _snap(frames, end_s)


def build_tracks_from_l0(
    media_facts_path: str | Path,
    frames_csv: str | Path,
    *,
    object_id: str,
    class_name: str | None = None,
    shot_id: str | None = None,
    shots_csv: str | Path | None = None,
    pts_start: float | None = None,
    pts_end: float | None = None,
    bbox_file: str | None = None,
    mask_dir: str | None = None,
    mask_index_csv: str | None = None,
    motion_file: str | None = None,
    per_frame_quality_csv: str | None = None,
    reference_asset_id: str | None = None,
    detector: dict[str, Any] | None = None,
    segmenter: dict[str, Any] | None = None,
    tracker: dict[str, Any] | None = None,
    mask_source: str | None = None,
    initialization_file: str | None = None,
) -> dict[str, Any]:
    """Build ``object_tracks.json`` directly from existing L0 artifacts."""
    l0 = load_l0(media_facts_path)
    frames = read_frames(frames_csv)
    start, end = resolve_window(frames, shot_id=shot_id, shots_csv=shots_csv,
                                pts_start=pts_start, pts_end=pts_end)

    return build_object_tracks(
        video_id=l0["video_id"],
        source_sha256=l0["source_sha256"],
        object_id=object_id,
        shot_id=shot_id,
        coded_size=l0["coded_size"],
        display_size=l0["display_size"],
        rotation_deg=l0["rotation_deg"],
        start_frame_no=start["frame_no"],
        start_pts_s=start["pts_s"],
        end_frame_no=end["frame_no"],
        end_pts_s=end["pts_s"],
        likely_variable_frame_rate=l0["likely_variable_frame_rate"],
        bbox_file=bbox_file or "tracks/bbox.npz",
        mask_dir=mask_dir or "masks",
        mask_index_csv=mask_index_csv or "masks/mask_index.csv",
        motion_file=motion_file or "tracks/motion_track.npz",
        per_frame_quality_csv=per_frame_quality_csv or "quality_per_frame.csv",
        class_name=class_name,
        reference_asset_id=reference_asset_id,
        detector=detector,
        segmenter=segmenter,
        tracker=tracker,
        mask_source=mask_source,
        initialization_file=initialization_file,
    )


def build_constraints_from_l0(
    media_facts_path: str | Path,
    frames_csv: str | Path,
    *,
    object_id: str,
    shot_id: str | None = None,
    shots_csv: str | Path | None = None,
    pts_start: float | None = None,
    pts_end: float | None = None,
    tracks_index: str = "object_tracks.json",
) -> dict[str, Any]:
    """Build ``object_constraints.json`` directly from existing L0 artifacts."""
    l0 = load_l0(media_facts_path)
    frames = read_frames(frames_csv)
    start, end = resolve_window(frames, shot_id=shot_id, shots_csv=shots_csv,
                                pts_start=pts_start, pts_end=pts_end)

    return build_object_constraints(
        video_id=l0["video_id"],
        source_sha256=l0["source_sha256"],
        object_id=object_id,
        coded_size=l0["coded_size"],
        display_size=l0["display_size"],
        rotation_deg=l0["rotation_deg"],
        start_frame_no=start["frame_no"],
        start_pts_s=start["pts_s"],
        end_frame_no=end["frame_no"],
        end_pts_s=end["pts_s"],
        tracks_index=tracks_index,
    )