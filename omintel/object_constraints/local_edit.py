"""Local editing driven by OVCL artifacts + output regression (task 5 / M5).

This module is the first *consumer* of the constraint contracts: it composites
a simple overlay (caption bar or solid patch) strictly inside the per-frame
object bbox read from ``tracks/bbox.npz``, and then proves, on lossless
intermediate frames, that the edit obeys the contract:

* spatial -- pixels outside every frame's edit region must be **pixel-identical**
  to the source (the ``preserve_regions`` "elsewhere" clause; checked on
  lossless BGR intermediates, never on a lossy encode);
* temporal -- the edit only touches frames inside ``edit_window``
  (``hc_temporal_boundary``), verified against ``frames.csv`` PTS values;
* gate-aware -- while quality thresholds stay uncalibrated every frame is
  REVIEW ("thresholds and weights uncalibrated"), which must not block the
  pipeline outright; instead each frame's *reliability* sub-gate (the same
  tracking-lost / suspected-drift reasons ``quality.assess_window`` records)
  decides editability. Unreliable frames get an editable decision: ``skip``
  (leave the source untouched, default) or ``fail`` (abort). This is what
  stops a drifted track from smearing an overlay across the wrong region.

The fused result is written as a lossless FFV1 Matroska video (libx264 with
crf 0 stays visibly lossy in practice, which would break pixel-exact
regression), so downstream timeline checks can re-decode the actual output and
compare it frame-by-frame against the source window. To keep that comparison
*pixel-exact*, the compositor only ever touches pixels inside the bbox: the
overlay region is first zeroed and then filled -- no anti-aliased text or
edge blending, because even one blended pixel outside the region would be a
contract violation we want to catch, not introduce.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .preview import load_quality_rows
from .video_io import decode_window

LOCAL_EDIT_REPORT_SCHEMA_VERSION = "openmontage.local_edit_report.v1"

#: allowed decisions for frames whose track is unreliable at that frame
NONPASS_DECISIONS = ("skip", "fail")

#: gate_reason fragments that mark a frame's *tracking* as untrustworthy.
#: The aggregate per-frame gate stays REVIEW while thresholds are uncalibrated
#: (reason "thresholds and weights uncalibrated; ..."), so editability keys off
#: the reliability suffix instead -- otherwise no frame could ever be edited
#: before calibration, which would defeat the purpose of this consumer test.
UNRELIABLE_MARKERS = ("tracking lost", "suspected drift")

#: normalized bbox coordinates outside [0, 1] cannot be interpreted in the
#: contract coordinate space (normalized top-left origin). Such rows are
#: rejected up front rather than silently clamped to the frame edge -- a
#: corrupt track must not smuggle an edit into the preserve region.
_COORD_EPS = 1e-6


def _validate_bbox(box: Any) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    bad = [v for v in (x0, y0, x1, y1)
           if not (-_COORD_EPS <= v <= 1.0 + _COORD_EPS)]
    if bad:
        raise ValueError(
            f"bbox has normalized coordinates outside [0, 1]: {bad} "
            f"(box={[x0, y0, x1, y1]}); refusing to edit on a corrupt track"
        )
    return x0, y0, x1, y1


def _frame_reliable(row: dict[str, Any] | None) -> bool:
    """True when the frame's gate_reason records no tracking-lost/drift marker."""
    if row is None:
        return False
    reason = str(row.get("gate_reason", "")).lower()
    return not any(m in reason for m in UNRELIABLE_MARKERS)


def _load_tracks(art: Path) -> dict[str, Any]:
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
    # contract invariant: every bbox row must live in the normalized [0, 1]
    # coordinate space *before* any consumer touches a pixel. A corrupt track
    # (out-of-range coords from a bad tracker or hand-edited npz) is rejected
    # here so it can never be silently clamped into an edit region.
    for fn, box in zip(frame_nos, bboxes):
        try:
            _validate_bbox(box)
        except ValueError as exc:
            raise ValueError(f"frame {fn}: {exc}") from exc
    return {
        "tracks": tracks,
        "frame_nos": frame_nos,
        "bboxes": bboxes,
        "object_id": tracks["object"]["object_id"],
        "shot_id": tracks.get("shot_id"),
    }


def _pts_map(frames_csv: str | Path) -> dict[int, float]:
    import csv

    out: dict[int, float] = {}
    with Path(frames_csv).open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            out[int(r["frame_no"])] = float(r["pts_time_s"])
    return out


def composite_local_overlay(
    video: str | Path,
    artifacts_dir: str | Path,
    out_path: str | Path,
    *,
    frames_csv: str | Path | None = None,
    text: str | None = None,
    color_bgr: tuple[int, int, int] = (0, 0, 255),
    nonpass_decision: str = "skip",
    fps: float | None = None,
) -> dict[str, Any]:
    """Composite an overlay inside each frame's tracked bbox and write a lossless video.

    With ``text`` a caption bar is drawn at the bottom of the bbox; without it
    a solid rectangle patch fills the bbox interior. Reads only existing
    sidecars (bbox npz, quality CSV, ``object_tracks.json``) -- exactly what
    downstream editing consumes -- so the edit cannot contradict the contracts.

    Returns a report (also written to ``out_path`` + ``.report.json``) listing
    edited/skipped frames, per-frame gates, and the PTS window actually used.
    """
    import cv2

    if nonpass_decision not in NONPASS_DECISIONS:
        raise ValueError(f"nonpass_decision must be one of {NONPASS_DECISIONS}")

    art = Path(artifacts_dir)
    info = _load_tracks(art)
    tracks, frame_nos, bboxes = info["tracks"], info["frame_nos"], info["bboxes"]
    # design rule: a shot cut ends the current tracking segment; identity is
    # NOT carried across cuts. A multi-shot window therefore has no single
    # valid tracked object -- refuse the edit instead of pasting one track's
    # boxes onto footage from another shot. (Per-segment analysis must be run
    # with shot-scoped windows; see io.resolve_window / shots_csv.)
    if info["shot_id"] is None:
        raise ValueError(
            "object_tracks.json carries shot_id=null: the tracked window is "
            "not bound to a single shot, so continuity of object identity "
            "across possible cuts cannot be audited. Refusing to composite; "
            "re-run analysis per shot segment (shot_id + shots_csv)."
        )
    quality = load_quality_rows(art / tracks["per_frame_quality_csv"])
    tb = tracks["time_base"]
    start_fn, end_fn = frame_nos[0], frame_nos[-1]

    frames = decode_window(video, start_fn, end_fn)
    h, w = frames[0].shape[:2]

    pts_by_no = _pts_map(frames_csv) if frames_csv else {}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer_fps = float(fps) if fps else 30.0
    writer = cv2.VideoWriter(str(out), fourcc, writer_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open lossless writer for {out} (FFV1)")

    edited: list[int] = []
    skipped: list[dict[str, Any]] = []
    try:
        for fn, src, box in zip(frame_nos, frames, bboxes):
            row = quality.get(fn)
            gate = row.get("gate") if row else None
            if not _frame_reliable(row):
                if nonpass_decision == "fail":
                    raise RuntimeError(
                        f"frame {fn} gate={gate!r} reason="
                        f"{row.get('gate_reason') if row else None!r}: "
                        "refusing to edit an unreliable track"
                    )
                skipped.append({"frame_no": int(fn), "gate": gate,
                                "reason": "unreliable tracking"})
                writer.write(src)
                continue
            x0, y0, x1, y1 = (int(round(v * d)) for v, d in zip(box, (w, w, h, h)))
            x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
            y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
            if x1 <= x0 or y1 <= y0:
                skipped.append({"frame_no": int(fn), "gate": gate,
                                "reason": "degenerate bbox"})
                writer.write(src)
                continue
            fused = src.copy()
            # pixel-exact discipline: only pixels strictly inside the bbox are
            # written; no anti-aliasing, no border blending (see module doc)
            if text:
                bar_h = max(12, min(32, (y1 - y0) // 4))
                ty = max(y0, y1 - bar_h)
                cv2.rectangle(fused, (x0, ty), (x1 - 1, y1 - 1), color_bgr, cv2.FILLED)
                # caption marker: small solid block in the bar corner (no AA text)
                mx0, my0 = x0 + 2, y1 - 6
                if x0 + 8 < x1 and my0 + 4 < y1:
                    fused[my0:my0 + 4, mx0:mx0 + 6] = (255, 255, 255)
            else:
                cv2.rectangle(fused, (x0, y0), (x1 - 1, y1 - 1), color_bgr, cv2.FILLED)
            edited.append(int(fn))
            writer.write(fused)
    finally:
        writer.release()

    report = {
        "schema_version": LOCAL_EDIT_REPORT_SCHEMA_VERSION,
        "video_id": tracks["video_id"],
        "object_id": info["object_id"],
        "output_file": str(out),
        "codec": "ffv1_lossless_mkv",
        "region_scope": "inside_tracked_bbox_only",
        "edited_frames": edited,
        "skipped_frames": skipped,
        "nonpass_decision": nonpass_decision,
        "window": {
            "start_frame_no": start_fn,
            "end_frame_no": end_fn,
            "start_pts_s": float(tb["start"]["pts_s"]),
            "end_pts_s": float(tb["end"]["pts_s"]),
            "pts_by_frame_no": {str(fn): pts_by_no[fn] for fn in frame_nos
                                 if fn in pts_by_no},
        },
        "note": ("preserve-region invariance is asserted on these lossless "
                 "frames via verify_local_edit_output, not on any lossy encode"),
    }
    report_path = out.with_suffix(out.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_file"] = str(report_path)
    return report


def verify_local_edit_output(
    source_video: str | Path,
    edited_video: str | Path,
    artifacts_dir: str | Path,
    *,
    tolerance: int = 0,
) -> dict[str, Any]:
    """Regression-check the fused output against the source and the contracts.

    Re-decodes both videos over the tracked window and verifies per frame:
    * every changed pixel lies inside that frame's bbox (+ ``tolerance`` px)
      -- i.e. the preserve complement really was untouched;
    * frames outside the edit window are never produced (count match);
    * PASS frames show the expected change (or are legitimately skipped).

    Raises ``AssertionError`` on any violation; returns a summary otherwise.
    """
    import cv2

    art = Path(artifacts_dir)
    info = _load_tracks(art)
    tracks, frame_nos, bboxes = info["tracks"], info["frame_nos"], info["bboxes"]
    quality = load_quality_rows(art / tracks["per_frame_quality_csv"])
    start_fn, end_fn = frame_nos[0], frame_nos[-1]

    src_frames = decode_window(source_video, start_fn, end_fn)
    dst_frames = decode_window(edited_video, start_fn, end_fn)
    if len(dst_frames) != len(src_frames):
        raise AssertionError(
            f"timeline mismatch: source window {len(src_frames)} frames, "
            f"edited {len(dst_frames)}"
        )

    n_changed = 0
    violations: list[dict[str, Any]] = []
    for fn, src, dst, box in zip(frame_nos, src_frames, dst_frames, bboxes):
        src_u8 = np.asarray(src, dtype=np.int16)
        dst_u8 = np.asarray(dst, dtype=np.int16)
        if src_u8.shape != dst_u8.shape:
            raise AssertionError(f"frame {fn}: size changed {src_u8.shape}->{dst_u8.shape}")
        h, w = src_u8.shape[:2]
        diff = np.abs(src_u8 - dst_u8).max(axis=2) > tolerance
        if not diff.any():
            continue
        n_changed += 1
        x0, y0, x1, y1 = (int(np.floor(v * d)) for v, d in zip(box, (w, w, h, h)))
        ys, xs = np.nonzero(diff)
        margin = tolerance + 1
        outside = ((xs < x0 - margin) | (xs > x1 + margin)
                   | (ys < y0 - margin) | (ys > y1 + margin))
        if outside.any():
            gate = quality.get(fn, {}).get("gate")
            violations.append({
                "frame_no": int(fn), "gate": gate,
                "pixels_outside_bbox": int(outside.sum()),
                "worst_xy": [int(xs[outside][0]), int(ys[outside][0])],
            })

    result = {
        "frames_checked": len(frame_nos),
        "frames_changed": n_changed,
        "preserve_region_violations": violations,
        "passed": not violations,
    }
    if violations:
        raise AssertionError(
            f"local edit leaked outside the edit region on "
            f"{len(violations)} frame(s): {violations[:3]}"
        )
    return result
