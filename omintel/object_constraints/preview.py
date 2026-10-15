"""Visual review preview: draw per-frame bbox + gate state onto a video (task 5).

The preview is a *review aid*, not a product artifact: it lets a human locate
anomalous frames quickly and jump back to ``initialization.json``'s init frame
to re-seed tracking. It reads only existing sidecars (bbox npz, quality CSV,
mask index CSV) plus the decoded source window, so it can never contradict the
contracts; every drawn overlay carries the frame's own ``frame_no`` so the
preview timeline is verifiable against ``frames.csv`` PTS values.

Gate colors: PASS green, REPAIR yellow, REVIEW red -- matching the three
states used by ``quality.assess_window`` / ``aggregate_gate``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

GATE_COLORS = {
    "PASS": (0, 200, 0),
    "REPAIR": (0, 215, 255),
    "REVIEW": (0, 0, 255),
}


def load_quality_rows(quality_csv: str | Path) -> dict[int, dict[str, Any]]:
    """Parse ``quality_per_frame.csv`` into ``{frame_no: row}``."""
    rows: dict[int, dict[str, Any]] = {}
    with Path(quality_csv).open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["frame_no"])] = r
    return rows


def render_preview(
    frames: Any,
    frame_nos: Any,
    bboxes: Any,
    quality_rows: dict[int, dict[str, Any]],
    out_path: str | Path,
    *,
    fps: float = 30.0,
    object_label: str = "",
) -> dict[str, Any]:
    """Draw each frame's box + gate banner and encode an MJPG preview video.

    Returns a summary with the written path and the list of flagged
    (non-PASS) frame numbers so reviewers can jump straight to them.
    ``frames[i]`` must correspond to ``frame_nos[i]`` (same indexing contract
    as ``decode_window``); mismatched lengths raise instead of drawing a
    misaligned preview.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - opencv-python is a repo dependency
        raise RuntimeError("opencv-python is required to render previews") from exc

    frames = list(frames)
    frame_nos = [int(f) for f in frame_nos]
    bboxes = np.asarray(bboxes, dtype=np.float64).reshape(len(bboxes), 4)
    if len(frames) != len(frame_nos) or len(bboxes) != len(frames):
        raise ValueError("frames/frame_nos/bboxes lengths must match")

    h, w = frames[0].shape[:2]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"),
                             float(fps), (w, h))
    if not writer.isOpened():  # pragma: no cover - depends on codec availability
        raise RuntimeError(f"cannot open VideoWriter for {out_path}")

    flagged: list[int] = []
    try:
        for frame, fn, box in zip(frames, frame_nos, bboxes):
            canvas = frame.copy()
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            row = quality_rows.get(fn, {})
            gate = row.get("gate", "REVIEW")
            color = GATE_COLORS.get(gate, GATE_COLORS["REVIEW"])
            if gate != "PASS":
                flagged.append(fn)
            x0, y0 = int(box[0] * w), int(box[1] * h)
            x1, y1 = int(box[2] * w), int(box[3] * h)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            reason = (row.get("gate_reason") or "")[:60]
            banner = f"f{fn} [{gate}] {object_label}".strip()
            cv2.putText(canvas, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2)
            if reason:
                cv2.putText(canvas, reason, (8, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1)
            writer.write(canvas)
    finally:
        writer.release()

    return {
        "preview_file": str(out_path),
        "frames_written": len(frames),
        "flagged_frames": flagged,
        "fps_used_for_preview": float(fps),
        "note": ("preview timing is wall-clock display only; authoritative "
                 "timeline stays frames.csv PTS via the contracts"),
    }


def render_preview_from_artifacts(
    video: str | Path,
    artifacts_dir: str | Path,
    out_path: str | Path | None = None,
    *,
    fps: float | None = None,
    object_label: str = "",
) -> dict[str, Any]:
    """Re-decode the tracked window and render its preview from sidecar files.

    Reads ``tracks/bbox.npz``, ``quality_per_frame.csv`` and the time window
    from ``object_tracks.json`` -- i.e. exactly what downstream editing would
    consume -- so the preview validates the artifact chain end to end.
    """
    import json

    from .video_io import decode_window, video_info

    art = Path(artifacts_dir)
    tracks = json.loads((art / "object_tracks.json").read_text(encoding="utf-8"))
    tb = tracks["time_base"]
    start_fn = int(tb["start"]["frame_no"])
    end_fn = int(tb["end"]["frame_no"])
    frame_nos = list(range(start_fn, end_fn + 1))

    bboxes = np.load(art / tracks["tracks"]["bbox_file"])["bboxes"]
    if len(bboxes) != len(frame_nos):
        raise ValueError(
            f"bbox.npz has {len(bboxes)} rows but window spans {len(frame_nos)} frames"
        )
    quality = load_quality_rows(art / tracks["per_frame_quality_csv"])

    frames = decode_window(video, start_fn, end_fn)
    if fps is None:
        fps = float(video_info(video)["fps"]) or 30.0
    target = Path(out_path) if out_path else art / "preview.mp4"
    result = render_preview(frames, frame_nos, bboxes, quality, target,
                            fps=fps, object_label=object_label)
    result["window"] = {"start_frame_no": start_fn, "end_frame_no": end_fn}
    return result
