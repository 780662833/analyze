from __future__ import annotations
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .utils import optional_float, parse_fraction, require_ffmpeg, run_cmd, sha256_file, tool_version, write_csv, write_json


def probe_json(args: list[str]) -> dict[str, Any]:
    cp = run_cmd(args)
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"无法解析 ffprobe JSON：{exc}") from exc


def extract_media_facts(video: Path, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """L0: Record immutable media facts and a canonical decoded-frame PTS timeline."""
    require_ffmpeg()
    video = video.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video.is_file():
        raise FileNotFoundError(f"视频文件不存在：{video}")

    info = probe_json(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)])
    streams = info.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError("源文件中没有可读取的视频流。")
    vs = video_streams[0]
    fmt = info.get("format", {})

    frame_info = probe_json([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time,pts_time,pkt_dts_time,pkt_duration_time,key_frame,pict_type",
        "-of", "json", str(video)
    ])
    frames: list[dict[str, Any]] = []
    for idx, fr in enumerate(frame_info.get("frames", [])):
        raw_pts = (fr.get("best_effort_timestamp_time") or fr.get("pts_time") or
                   fr.get("pkt_pts_time") or fr.get("pkt_dts_time"))
        pts = optional_float(raw_pts)
        frames.append({
            "frame_no": idx,
            "pts_time_s": pts,
            "pts_time_raw": raw_pts,
            "pkt_duration_s": optional_float(fr.get("pkt_duration_time")),
            "key_frame": int(fr.get("key_frame", 0) or 0),
            "pict_type": fr.get("pict_type"),
        })
    if not frames:
        raise RuntimeError("ffprobe 未能获取视频帧时间戳；请确认视频可解码。")

    # Only adjacent decoded frames with valid PTS are used for interval diagnostics.
    adjacent_diffs = []
    missing_pts = 0
    non_monotonic_pts = 0
    for left, right in zip(frames, frames[1:]):
        a, b = left["pts_time_s"], right["pts_time_s"]
        if a is None or b is None:
            continue
        delta = float(b) - float(a)
        adjacent_diffs.append(delta)
        if delta <= 0:
            non_monotonic_pts += 1
    missing_pts = sum(1 for r in frames if r["pts_time_s"] is None)
    positive_diffs = [d for d in adjacent_diffs if d > 0]
    median_interval = statistics.median(positive_diffs) if positive_diffs else None
    interval_cv = None
    if len(positive_diffs) >= 2:
        mean = sum(positive_diffs) / len(positive_diffs)
        variance = sum((x - mean) ** 2 for x in positive_diffs) / len(positive_diffs)
        interval_cv = math.sqrt(variance) / mean if mean else None

    format_duration = optional_float(fmt.get("duration"))
    stream_duration = optional_float(vs.get("duration"))
    duration = format_duration or stream_duration
    valid_pts = [float(r["pts_time_s"]) for r in frames if r["pts_time_s"] is not None]
    if duration is None and valid_pts:
        duration = valid_pts[-1] - valid_pts[0] + (median_interval or 0.0)

    rotation = None
    for side in vs.get("side_data_list", []) or []:
        if side.get("rotation") is not None:
            rotation = optional_float(side.get("rotation"))
            break
    if rotation is None:
        rotation = optional_float((vs.get("tags") or {}).get("rotate"))
    width, height = int(vs.get("width", 0) or 0), int(vs.get("height", 0) or 0)
    rot_norm = int(round(rotation or 0)) % 360
    display_width, display_height = (height, width) if rot_norm in (90, 270) else (width, height)

    write_csv(out_dir / "frames.csv", frames, ["frame_no", "pts_time_s", "pts_time_raw", "pkt_duration_s", "key_frame", "pict_type"])
    facts = {
        "schema_version": "openmontage.media_facts.v1",
        "pipeline_version": f"openmontage-intelligence/{__version__}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(video), "file_name": video.name, "size_bytes": video.stat().st_size, "sha256": sha256_file(video)},
        "tools": {"ffmpeg": tool_version("ffmpeg"), "ffprobe": tool_version("ffprobe")},
        "container": {
            "format_name": fmt.get("format_name"), "format_long_name": fmt.get("format_long_name"),
            "start_time_s": optional_float(fmt.get("start_time")), "duration_s": duration,
            "bit_rate": optional_float(fmt.get("bit_rate")),
        },
        "video_stream": {
            "index": vs.get("index"), "codec_name": vs.get("codec_name"), "profile": vs.get("profile"),
            "width": width, "height": height, "display_width": display_width, "display_height": display_height,
            "rotation_deg": rotation, "pix_fmt": vs.get("pix_fmt"), "time_base": vs.get("time_base"),
            "r_frame_rate": vs.get("r_frame_rate"), "avg_frame_rate": vs.get("avg_frame_rate"),
            "avg_frame_rate_numeric": parse_fraction(vs.get("avg_frame_rate")),
            "r_frame_rate_numeric": parse_fraction(vs.get("r_frame_rate")),
            "bit_rate": optional_float(vs.get("bit_rate")), "nb_frames_reported": vs.get("nb_frames"),
        },
        "other_streams": [
            {k: s.get(k) for k in ("index", "codec_type", "codec_name", "sample_rate", "channels", "channel_layout", "duration", "bit_rate") if k in s}
            for s in streams if s.get("index") != vs.get("index")
        ],
        "frame_timeline": {
            "frame_count_decoded": len(frames), "frames_with_pts": len(valid_pts),
            "missing_pts_count": missing_pts, "non_monotonic_pts_count": non_monotonic_pts,
            "first_pts_time_s": valid_pts[0] if valid_pts else None,
            "last_pts_time_s": valid_pts[-1] if valid_pts else None,
            "median_positive_frame_interval_s": median_interval,
            "estimated_fps_from_pts": 1.0 / median_interval if median_interval else None,
            "frame_interval_coefficient_of_variation": interval_cv,
            "likely_variable_frame_rate": bool(interval_cv is not None and interval_cv > 0.02),
            "canonical_time_source": "ffprobe frame best_effort_timestamp_time with pts_time/pkt_pts_time/pkt_dts_time fallbacks",
        },
        "original_probe": {"format_keys": sorted(fmt.keys()), "video_stream_keys": sorted(vs.keys())},
    }
    write_json(out_dir / "media_facts.json", facts)
    return facts, frames
