from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .detectors import detect_ffmpeg_scdet, detect_opencv_fallback, detect_pyscenedetect, detect_transnetv2
from .fusion import build_evidence_pack, fuse_candidates, write_timeline_outputs
from .media_facts import extract_media_facts
from .utils import write_json


def write_report(out_dir: Path, facts: dict[str, Any], a: list[dict[str, Any]], b: list[dict[str, Any]],
                 fused: list[dict[str, Any]], args: argparse.Namespace, actual_a: str, actual_b: str,
                 evidence: dict[str, Any]) -> None:
    counts = {status: sum(1 for r in fused if r["status"] == status)
              for status in ("agreement", "scene_a_only", "scene_b_only")}
    timeline = facts["frame_timeline"]
    lines = [
        "# OpenMontage Intelligence Run Report", "",
        f"- Pipeline version: `{__version__}`", f"- Completed UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Source: `{facts['source']['file_name']}`", f"- SHA-256: `{facts['source']['sha256']}`",
        f"- Duration (s): `{facts['container']['duration_s']}`",
        f"- Decoded frames: `{timeline['frame_count_decoded']}`",
        f"- Resolution: `{facts['video_stream']['width']}x{facts['video_stream']['height']}`",
        f"- Estimated FPS from PTS: `{timeline['estimated_fps_from_pts']}`",
        f"- Likely VFR: `{timeline['likely_variable_frame_rate']}`",
        f"- Missing PTS: `{timeline['missing_pts_count']}`; non-monotonic adjacent PTS: `{timeline['non_monotonic_pts_count']}`",
        "", "## Detector setup", "",
        f"- Detector A actual: `{actual_a}`; threshold={args.a_threshold}; min gap={args.min_gap_s}s",
        f"- Detector B actual: `{actual_b}`; threshold={args.b_threshold if actual_b == 'ffmpeg_scdet' else 'model-native' }",
        f"- Fusion tolerance: `{args.match_tolerance_s}` s",
        "", "## Candidates", "",
        f"- A candidates: **{len(a)}**", f"- B candidates: **{len(b)}**",
        f"- Agreement (corroboration only): **{counts['agreement']}**",
        f"- A-only: **{counts['scene_a_only']}**", f"- B-only: **{counts['scene_b_only']}**",
        f"- Evidence events: **{evidence.get('event_count', 0)}**",
        f"- Complete evidence packs: **{sum(1 for e in evidence.get('events', []) if e.get('evidence_status') == 'complete')}**",
        "", "## Outputs", "",
        "- `media_facts.json`, `frames.csv`: source identity, stream metadata and canonical PTS timeline.",
        "- `scene_a_*.csv`, `scene_b_*.csv`: original detector outputs; do not discard them after fusion.",
        "- `shot_candidates.csv/json`: fused shot-boundary candidates, with source provenance.",
        "- `provisional_shots.csv`: provisional temporal partition, not manually validated shots.",
        "- `events.json`, `evidence_index.json`, `evidence/<event_id>/*.jpg`: evidence-linked events with before/at/after frames and hashes.",
        "", "## Interpretation limits", "",
        "1. `agreement` only means detectors produced candidates within the configured time tolerance; it does not mean the boundary is correct.",
        "2. OpenCV fallback is a test path, not equivalent to PySceneDetect. It is labeled explicitly in metadata.",
        "3. FFmpeg scdet, PySceneDetect ContentDetector and TransNetV2 scores/thresholds are not numerically comparable.",
        "4. `provisional_shots.csv` and `events.json` are boundary analysis outputs; action semantics and creative intent are not inferred in this phase.",
        "5. Precision/Recall/F1 cannot be reported until a human-reviewed ground-truth set is available.",
    ]
    if actual_a == "opencv_hsv_content_fallback":
        lines += ["", "> **TEST PATH:** Detector A used OpenCV HSV fallback. Re-run with PySceneDetect installed for formal A/B comparison."]
    if actual_b == "ffmpeg_scdet":
        lines += ["", "> Detector B used FFmpeg scdet. The optional TransNetV2 adapter is available when its official inference package and compatible weights are installed."]
    (out_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    video = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[L0] 建立媒体事实与 PTS 时间轴…", flush=True)
    facts, frames = extract_media_facts(video, out_dir)
    print(f"[L0] 完成：{len(frames)} 帧；SHA256={facts['source']['sha256'][:16]}…", flush=True)

    print(f"[L1-A] 镜头边界检测：{args.scene_a}…", flush=True)
    if args.scene_a == "pyscenedetect":
        a = detect_pyscenedetect(video, frames, out_dir, args.a_threshold, args.min_gap_s)
        actual_a = "pyscenedetect_content"
    else:
        a = detect_opencv_fallback(video, frames, out_dir, args.a_threshold, args.min_gap_s)
        actual_a = "opencv_hsv_content_fallback"
    print(f"[L1-A] 候选：{len(a)}", flush=True)

    print(f"[L1-B] 镜头边界检测：{args.scene_b}…", flush=True)
    if args.scene_b == "ffmpeg-scdet":
        b = detect_ffmpeg_scdet(video, frames, out_dir, args.b_threshold, args.min_gap_s)
        actual_b = "ffmpeg_scdet"
    else:
        b = detect_transnetv2(video, frames, out_dir)
        actual_b = "transnetv2"
    print(f"[L1-B] 候选：{len(b)}", flush=True)

    print("[L2] 融合候选、构建暂定镜头表与证据包…", flush=True)
    fused = fuse_candidates(a, b, args.match_tolerance_s)
    write_timeline_outputs(out_dir, fused, facts)
    evidence = build_evidence_pack(video, out_dir, fused, frames, facts)
    write_report(out_dir, facts, a, b, fused, args, actual_a, actual_b, evidence)
    run_info = {
        "pipeline_version": __version__, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": facts["source"]["sha256"], "out_dir": str(out_dir),
        "detector_a": actual_a, "detector_b": actual_b,
        "detector_a_candidates": len(a), "detector_b_candidates": len(b), "fused_candidates": len(fused),
        "evidence_events": evidence["event_count"],
        "note": "Pipeline run completion is not equivalent to ground-truth accuracy validation.",
    }
    write_json(out_dir / "analysis_run.json", run_info)
    print(f"[DONE] {out_dir}", flush=True)
    print("      " + ", ".join(f"{k}={v}" for k, v in {
        "agreement": sum(1 for x in fused if x["status"] == "agreement"),
        "A_only": sum(1 for x in fused if x["status"] == "scene_a_only"),
        "B_only": sum(1 for x in fused if x["status"] == "scene_b_only"),
        "evidence_events": evidence["event_count"],
    }.items()), flush=True)
    return run_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenMontage Intelligence: L0 media facts, L1 dual scene detection, L2 evidence fusion")
    parser.add_argument("--video", required=True, help="本地源视频路径")
    parser.add_argument("--out", default="analysis_run", help="输出目录")
    parser.add_argument("--scene-a", choices=["pyscenedetect", "opencv-fallback"], default="pyscenedetect",
                        help="A 路；opencv-fallback 只用于管线自测")
    parser.add_argument("--scene-b", choices=["ffmpeg-scdet", "transnetv2"], default="ffmpeg-scdet",
                        help="B 路；TransNetV2 需预先安装官方推理模块、依赖与权重")
    parser.add_argument("--a-threshold", type=float, default=27.0, help="PySceneDetect/fallback threshold")
    parser.add_argument("--b-threshold", type=float, default=10.0, help="FFmpeg scdet threshold, range 0-100")
    parser.add_argument("--min-gap", dest="min_gap_s", type=float, default=0.30, help="同一路候选的最小间隔（秒）")
    parser.add_argument("--match-tolerance", dest="match_tolerance_s", type=float, default=0.10,
                        help="跨检测器候选匹配容差（秒）")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.min_gap_s <= 0 or args.match_tolerance_s < 0:
        parser.error("--min-gap 必须 > 0；--match-tolerance 必须 >= 0")
    try:
        run_pipeline(args)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
