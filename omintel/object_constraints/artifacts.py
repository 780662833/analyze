"""Writers for the per-frame sidecar files referenced by the contracts.

The JSON contracts are index-only; the heavy per-frame data lives in these
sidecars:

- ``masks/frame_XXXXX.png``    -- binary per-frame masks (uint8, 0/255)
- ``masks/mask_index.csv``     -- frame_no -> pts_time_s -> mask file
- ``tracks/bbox.npz``          -- per-frame normalized bboxes ``(n, 4)``
- ``tracks/motion_track.npz``  -- ``points``/``visibility``/``confidence``
- ``quality_per_frame.csv``    -- per-frame scalar states + gate

All functions write to ``out_dir`` and return the relative sidecar paths the
contract's ``tracks`` block should reference.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MASK_INDEX_FIELDS = ("frame_no", "pts_time_s", "mask_file")
QUALITY_FIELDS = (
    "frame_no", "pts_time_s", "mask_area_ratio_change",
    "flow_agreement", "feature_loss_ratio", "anomaly_score", "gate", "gate_reason",
)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_masks(
    mask_dir: Path,
    masks: Sequence[Any],
    frame_nos: Sequence[int],
    pts_ts: Sequence[float],
) -> list[dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - opencv-python is a repo dependency
        raise RuntimeError("opencv-python is required to write PNG masks") from exc

    mask_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for fn, pts, mask in zip(frame_nos, pts_ts, masks):
        name = f"frame_{int(fn):05d}.png"
        img = (np.asarray(mask).astype(bool)).astype(np.uint8) * 255
        cv2.imwrite(str(mask_dir / name), img)
        index.append({"frame_no": int(fn), "pts_time_s": pts, "mask_file": f"masks/{name}"})
    return index


def write_bbox_npz(path: Path, bboxes: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, bboxes=np.asarray(bboxes, dtype=np.float32).reshape(len(bboxes), 4))


def write_motion_npz(path: Path, motion: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        points=np.asarray(motion["points"], dtype=np.float32),
        visibility=np.asarray(motion["visibility"], dtype=np.float32),
        confidence=np.asarray(motion["confidence"], dtype=np.float32),
    )


def write_sidecars(
    out_dir: Path,
    *,
    frame_nos: Sequence[int],
    pts_ts: Sequence[float],
    masks: Sequence[Any],
    bboxes: Sequence[Sequence[float]],
    motion: dict[str, Any],
    quality_rows: Sequence[dict[str, Any]],
) -> dict[str, str]:
    """Write all sidecars and return the relative paths for the contract."""
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_index = write_masks(out_dir / "masks", masks, frame_nos, pts_ts)
    _write_csv(out_dir / "masks" / "mask_index.csv", MASK_INDEX_FIELDS, mask_index)

    write_bbox_npz(out_dir / "tracks" / "bbox.npz", bboxes)
    write_motion_npz(out_dir / "tracks" / "motion_track.npz", motion)

    _write_csv(out_dir / "quality_per_frame.csv", QUALITY_FIELDS, quality_rows)

    return {
        "mask_dir": "masks",
        "mask_index_csv": "masks/mask_index.csv",
        "bbox_file": "tracks/bbox.npz",
        "motion_file": "tracks/motion_track.npz",
        "per_frame_quality_csv": "quality_per_frame.csv",
    }