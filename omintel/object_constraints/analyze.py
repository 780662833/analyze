"""Orchestration entry: L0 artifacts -> object contracts (v1).

This is the first milestone of the object-constraint stage: it consumes the
existing L0 artifacts and writes the two stable contracts. Detector/segmenter/
tracker adapters plug in later; v1 only wires the deterministic plumbing so the
contracts can be produced and validated end to end.

Run as ``python -m omintel.object_constraints.analyze`` or import ``run_analysis``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from omintel.utils import write_json

from .artifacts import write_sidecars
from .detector import BBox, StaticBoxDetector
from .io import (
    build_constraints_from_l0,
    build_tracks_from_l0,
    load_l0,
    read_frames,
    resolve_window,
)
from .quality import aggregate_gate, assess_window
from .schemas import validate_object_constraints, validate_object_tracks
from .segmenter import BoxMaskSegmenter
from .tracker import NullPointTracker
from .video_io import decode_window, video_info


def run_analysis(
    *,
    media_facts: str | Path,
    frames: str | Path,
    out_dir: str | Path,
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
) -> dict[str, Any]:
    """Load L0, resolve the edit window, build + validate + write both contracts."""
    # Force-load now so input errors surface before any output is written.
    load_l0(media_facts)
    read_frames(frames)

    tracks = build_tracks_from_l0(
        media_facts, frames,
        object_id=object_id,
        class_name=class_name,
        shot_id=shot_id,
        shots_csv=shots_csv,
        pts_start=pts_start,
        pts_end=pts_end,
        bbox_file=bbox_file,
        mask_dir=mask_dir,
        mask_index_csv=mask_index_csv,
        motion_file=motion_file,
        per_frame_quality_csv=per_frame_quality_csv,
        reference_asset_id=reference_asset_id,
    )
    validate_object_tracks(tracks)

    constraints = build_constraints_from_l0(
        media_facts, frames,
        object_id=object_id,
        shot_id=shot_id,
        shots_csv=shots_csv,
        pts_start=pts_start,
        pts_end=pts_end,
    )
    validate_object_constraints(constraints)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "object_tracks.json", tracks)
    write_json(out / "object_constraints.json", constraints)

    return {
        "video_id": tracks["video_id"],
        "object_id": object_id,
        "time_window": tracks["time_base"],
        "wrote": [
            str(out / "object_tracks.json"),
            str(out / "object_constraints.json"),
        ],
    }


def _window_rows(frames: list[dict[str, Any]], start_frame_no: int, end_frame_no: int) -> list[dict[str, Any]]:
    by_no = {r["frame_no"]: r for r in frames}
    missing = [fn for fn in range(start_frame_no, end_frame_no + 1) if fn not in by_no]
    if missing:
        raise ValueError(f"frames.csv missing frame_nos in window: {missing[:5]}")
    return [by_no[fn] for fn in range(start_frame_no, end_frame_no + 1)]


def _seed_points_from_bbox(box: BBox, coded_size: tuple[int, int]) -> list[list[float]]:
    w, h = max(1, coded_size[0]), max(1, coded_size[1])
    cx = (box.x_min + box.x_max) / 2.0 * w
    cy = (box.y_min + box.y_max) / 2.0 * h
    return [[cx, cy]]


def _provenance(adapter: Any, **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"backend": type(adapter).__name__, "model_version": None, "params": {}}
    data.update(extra)
    return data


def run_frames(
    frames: Any,
    *,
    media_facts: str | Path,
    frames_csv: str | Path,
    out_dir: str | Path,
    object_id: str,
    class_name: str | None = None,
    shot_id: str | None = None,
    shots_csv: str | Path | None = None,
    pts_start: float | None = None,
    pts_end: float | None = None,
    bbox: BBox | None = None,
    detector: Any = None,
    segmenter: Any = None,
    tracker: Any = None,
    reference_asset_id: str | None = None,
) -> dict[str, Any]:
    """Model-free closed loop: frames -> per-frame data -> sidecars + contracts.

    ``frames`` are the decoded frames for the edit window, in frame order, with
    length equal to the window's frame count. Any adapter can be injected; the
    defaults are deterministic stubs (static box detector, box-mask segmenter,
    null tracker) so the whole loop runs without a deep-learning model.
    """
    l0 = load_l0(media_facts)
    fcsv = read_frames(frames_csv)
    start, end = resolve_window(fcsv, shot_id=shot_id, shots_csv=shots_csv,
                                pts_start=pts_start, pts_end=pts_end)
    window = _window_rows(fcsv, start["frame_no"], end["frame_no"])
    if len(frames) != len(window):
        raise ValueError(f"expected {len(window)} frames for the window, got {len(frames)}")
    frame_nos = [r["frame_no"] for r in window]
    pts_ts = [r["pts_time_s"] for r in window]

    detector = detector or StaticBoxDetector(bbox or BBox(0.2, 0.3, 0.4, 0.5))
    segmenter = segmenter or BoxMaskSegmenter()
    tracker = tracker or NullPointTracker()

    key_idx = len(frames) // 2
    boxes = detector.detect(frames[key_idx], [class_name or object_id])
    if not boxes:
        raise ValueError("detector returned no boxes")
    box = boxes[0]

    state = segmenter.seed(frames[key_idx], box)
    masks = [segmenter.propagate(state, f) for f in frames]

    seed_points = _seed_points_from_bbox(box, l0["coded_size"])
    motion = tracker.track(frames, seed_points, (0, len(frames) - 1))
    quality_rows = assess_window(frame_nos, pts_ts, masks, motion)

    out = Path(out_dir)
    side = write_sidecars(
        out,
        frame_nos=frame_nos,
        pts_ts=pts_ts,
        masks=masks,
        bboxes=[list(box)] * len(frames),
        motion=motion,
        quality_rows=quality_rows,
    )

    tracks = build_tracks_from_l0(
        media_facts, frames_csv,
        object_id=object_id,
        class_name=class_name,
        shot_id=shot_id,
        shots_csv=shots_csv,
        pts_start=pts_start,
        pts_end=pts_end,
        bbox_file=side["bbox_file"],
        mask_dir=side["mask_dir"],
        mask_index_csv=side["mask_index_csv"],
        motion_file=side["motion_file"],
        per_frame_quality_csv=side["per_frame_quality_csv"],
        reference_asset_id=reference_asset_id,
        detector=_provenance(detector, prompt_texts=[class_name or object_id]),
        segmenter=_provenance(segmenter),
        tracker=_provenance(tracker),
    )
    validate_object_tracks(tracks)

    constraints = build_constraints_from_l0(
        media_facts, frames_csv,
        object_id=object_id,
        shot_id=shot_id,
        shots_csv=shots_csv,
        pts_start=pts_start,
        pts_end=pts_end,
    )
    validate_object_constraints(constraints)

    write_json(out / "object_tracks.json", tracks)
    write_json(out / "object_constraints.json", constraints)

    return {
        "video_id": tracks["video_id"],
        "object_id": object_id,
        "time_window": tracks["time_base"],
        "gate": aggregate_gate(quality_rows),
        "wrote": [
            str(out / "object_tracks.json"),
            str(out / "object_constraints.json"),
        ],
        "sidecars": side,
    }


def run_video(
    video: str | Path,
    *,
    media_facts: str | Path,
    frames_csv: str | Path,
    out_dir: str | Path,
    object_id: str,
    class_name: str | None = None,
    shot_id: str | None = None,
    shots_csv: str | Path | None = None,
    pts_start: float | None = None,
    pts_end: float | None = None,
    bbox: BBox | None = None,
    detector: Any = None,
    segmenter: Any = None,
    tracker: Any = None,
    reference_asset_id: str | None = None,
) -> dict[str, Any]:
    """Decode real frames via ``cv2.VideoCapture``, then run the closed loop.

    Frame-index mapping stays explicit: ``decode_window`` guarantees
    ``frames[i]`` corresponds to ``window.frame_no = start + i``; the remaining
    pipeline is the same model-free loop as ``run_frames``. The decoder-side
    size and its match against the ffprobe coded size are reported (not
    enforced) so a rotation/resize mismatch surfaces instead of corrupting
    coordinates silently.
    """
    l0 = load_l0(media_facts)
    fcsv = read_frames(frames_csv)
    start, end = resolve_window(fcsv, shot_id=shot_id, shots_csv=shots_csv,
                                pts_start=pts_start, pts_end=pts_end)

    info = video_info(video)
    frames = decode_window(video, start["frame_no"], end["frame_no"])

    result = run_frames(
        frames,
        media_facts=media_facts,
        frames_csv=frames_csv,
        out_dir=out_dir,
        object_id=object_id,
        class_name=class_name,
        shot_id=shot_id,
        shots_csv=shots_csv,
        pts_start=pts_start,
        pts_end=pts_end,
        bbox=bbox,
        detector=detector,
        segmenter=segmenter,
        tracker=tracker,
        reference_asset_id=reference_asset_id,
    )
    coded_w, coded_h = l0["coded_size"]
    result["video_info"] = info
    result["decoded_size"] = {"width": int(frames[0].shape[1]), "height": int(frames[0].shape[0])}
    result["coded_size"] = {"width": int(coded_w), "height": int(coded_h)}
    result["decoded_size_matches_coded"] = (
        int(frames[0].shape[1]) == int(coded_w)
        and int(frames[0].shape[0]) == int(coded_h)
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenMontage object constraints: L0 artifacts -> object contracts (v1)"
    )
    parser.add_argument("--media-facts", required=True, help="L0 产物 media_facts.json")
    parser.add_argument("--frames", required=True, help="L0 产物 frames.csv")
    parser.add_argument("--video", help="真实视频路径；提供后走 cv2.VideoCapture 解码再运行闭环")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--object-id", required=True, help="对象 ID，如 shoe_01")
    parser.add_argument("--class", dest="class_name", help="对象类别，如 sneaker")
    parser.add_argument("--shot-id", help="从 provisional_shots.csv 选取镜头")
    parser.add_argument("--shots-csv", help="provisional_shots.csv 路径")
    parser.add_argument("--start", dest="pts_start", type=float, help="起始 PTS（秒）")
    parser.add_argument("--end", dest="pts_end", type=float, help="结束 PTS（秒）")
    parser.add_argument("--bbox-file", help="bbox 落盘路径覆盖")
    parser.add_argument("--mask-dir", help="掩码目录覆盖")
    parser.add_argument("--mask-index-csv", help="掩码索引 CSV 覆盖")
    parser.add_argument("--motion-file", help="运动轨迹文件覆盖")
    parser.add_argument("--quality-csv", dest="per_frame_quality_csv", help="逐帧质量 CSV 覆盖")
    parser.add_argument("--reference-asset-id", help="参考商品资产 ID")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.video:
            result = run_video(
                video=args.video,
                media_facts=args.media_facts,
                frames_csv=args.frames,
                out_dir=args.out,
                object_id=args.object_id,
                class_name=args.class_name,
                shot_id=args.shot_id,
                shots_csv=args.shots_csv,
                pts_start=args.pts_start,
                pts_end=args.pts_end,
            )
        else:
            result = run_analysis(
                media_facts=args.media_facts,
                frames=args.frames,
                out_dir=args.out,
                object_id=args.object_id,
                class_name=args.class_name,
                shot_id=args.shot_id,
                shots_csv=args.shots_csv,
                pts_start=args.pts_start,
                pts_end=args.pts_end,
                bbox_file=args.bbox_file,
                mask_dir=args.mask_dir,
                mask_index_csv=args.mask_index_csv,
                motion_file=args.motion_file,
                per_frame_quality_csv=args.per_frame_quality_csv,
                reference_asset_id=args.reference_asset_id,
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())