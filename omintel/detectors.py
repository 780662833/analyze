from __future__ import annotations
import re
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

from .utils import optional_float, require_ffmpeg, run_cmd, tool_version, write_csv, write_json


def frame_pts(frame_no: int, frames: list[dict[str, Any]]) -> float | None:
    if 0 <= frame_no < len(frames):
        return frames[frame_no].get("pts_time_s")
    return None


def suppress_candidates(rows: list[dict[str, Any]], min_gap_s: float) -> list[dict[str, Any]]:
    """Non-maximum suppression of adjacent candidate frames; keep the highest score."""
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True):
        t = float(row["time_s"])
        if all(abs(t - float(k["time_s"])) >= min_gap_s for k in kept):
            kept.append(row)
    return sorted(kept, key=lambda r: float(r["time_s"]))


def detect_pyscenedetect(video: Path, frames: list[dict[str, Any]], out_dir: Path,
                         threshold: float = 27.0, min_gap_s: float = 0.30) -> list[dict[str, Any]]:
    try:
        import scenedetect  # type: ignore
        from scenedetect import ContentDetector, SceneManager, open_video  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "未安装 PySceneDetect。请在联网的开发环境运行：python -m pip install 'scenedetect[opencv]>=0.7,<0.8'。"
            "临时测试可用 --scene-a opencv-fallback，但不能将 fallback 结果称为 PySceneDetect 结果。"
        ) from exc
    stream = open_video(str(video))
    rows: list[dict[str, Any]] = []
    try:
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=f"{min_gap_s}s"))
        manager.detect_scenes(video=stream, show_progress=False)
        scenes = manager.get_scene_list(start_in_scene=True)
        for index, scene in enumerate(scenes[1:], start=1):
            tc = scene[0]
            fn = int(tc.get_frames())
            t = frame_pts(fn, frames)
            if t is None:
                t = float(tc.get_seconds())
            rows.append({
                "boundary_id": f"A{index:04d}", "frame_no": fn, "time_s": round(float(t), 9),
                "detector": "pyscenedetect_content", "score": None, "threshold": threshold,
                "min_scene_len_s": min_gap_s, "detector_time_s": round(float(tc.get_seconds()), 9),
            })
        version = getattr(scenedetect, "__version__", "unknown")
    finally:
        close = getattr(stream, "close", None) or getattr(stream, "release", None)
        if callable(close):
            close()
    write_csv(out_dir / "scene_a_pyscenedetect.csv", rows,
              ["boundary_id", "frame_no", "time_s", "detector", "score", "threshold", "min_scene_len_s", "detector_time_s"])
    write_json(out_dir / "scene_a_metadata.json", {
        "detector": "PySceneDetect ContentDetector", "version": version,
        "threshold": threshold, "min_scene_len_s": min_gap_s,
        "note": "scene cut frame indexes mapped to canonical ffprobe timeline when available",
    })
    return rows


def detect_opencv_fallback(video: Path, frames: list[dict[str, Any]], out_dir: Path,
                           threshold: float = 27.0, min_gap_s: float = 0.30) -> list[dict[str, Any]]:
    if cv2 is None:
        raise RuntimeError("OpenCV 未安装。请运行 python -m pip install opencv-python。")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频：{video}")
    scores: list[dict[str, Any]] = []
    previous = None
    index = 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        sample = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA) if bgr.shape[1] > 320 else bgr
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype("float32")
        score = 0.0 if previous is None else float(abs(hsv - previous).mean())
        t = frame_pts(index, frames)
        if t is None:
            t = index / max(fps, 1e-6)
        scores.append({"frame_no": index, "time_s": float(t), "score": score})
        previous = hsv
        index += 1
    cap.release()
    raw = [{
        "boundary_id": "", "frame_no": int(r["frame_no"]), "time_s": float(r["time_s"]),
        "detector": "opencv_hsv_content_fallback", "score": float(r["score"]),
        "threshold": threshold, "min_scene_len_s": min_gap_s, "detector_time_s": float(r["time_s"]),
    } for r in scores[1:] if float(r["score"]) >= threshold]
    peaks = suppress_candidates(raw, min_gap_s)
    for index, row in enumerate(peaks, 1):
        row["boundary_id"] = f"A{index:04d}"
    write_csv(out_dir / "scene_a_opencv_fallback.csv", peaks,
              ["boundary_id", "frame_no", "time_s", "detector", "score", "threshold", "min_scene_len_s", "detector_time_s"])
    write_csv(out_dir / "scene_a_fallback_frame_scores.csv", scores, ["frame_no", "time_s", "score"])
    write_json(out_dir / "scene_a_metadata.json", {
        "detector": "OpenCV HSV mean absolute difference", "mode": "FALLBACK ONLY",
        "threshold": threshold, "min_scene_len_s": min_gap_s,
        "warning": "Not a PySceneDetect result; use only to test pipeline wiring.",
    })
    return peaks


def parse_ffmpeg_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    frame_re = re.compile(r"^frame:\s*(\d+)\s+pts:\S+\s+pts_time:([^\s]+)")
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            match = frame_re.match(line)
            if match:
                if current is not None:
                    rows.append(current)
                current = {"frame_no": int(match.group(1)), "time_s": optional_float(match.group(2)),
                           "mafd": None, "score": None, "scd_time": None}
                continue
            if current is None or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "lavfi.scd.score":
                current["score"] = optional_float(value)
            elif key == "lavfi.scd.mafd":
                current["mafd"] = optional_float(value)
            elif key == "lavfi.scd.time":
                current["scd_time"] = value
    if current is not None:
        rows.append(current)
    return rows


def detect_ffmpeg_scdet(video: Path, frames: list[dict[str, Any]], out_dir: Path,
                        threshold: float = 10.0, min_gap_s: float = 0.30) -> list[dict[str, Any]]:
    require_ffmpeg()
    metadata_name = "ffmpeg_frame_metadata.txt"
    # No shell is used. Relative metadata path prevents Windows drive-colon escaping issues.
    filter_expr = f"scdet=threshold={threshold}:sc_pass=0,metadata=mode=print:file={metadata_name}"
    run_cmd(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error", "-i", str(video.resolve()),
             "-vf", filter_expr, "-an", "-f", "null", "-"], cwd=out_dir)
    meta_path = out_dir / metadata_name
    if not meta_path.exists():
        raise RuntimeError("FFmpeg 未生成 scdet 逐帧评分；请检查 FFmpeg 是否支持 scdet 和 metadata 滤镜。")
    scores = parse_ffmpeg_metadata(meta_path)
    for row in scores:
        pts = frame_pts(int(row["frame_no"]), frames)
        if pts is not None:
            row["time_s"] = pts
    raw = []
    for row in scores[1:]:
        score = row.get("score")
        if score is not None and float(score) >= threshold and row.get("time_s") is not None:
            raw.append({
                "boundary_id": "", "frame_no": int(row["frame_no"]), "time_s": float(row["time_s"]),
                "detector": "ffmpeg_scdet", "score": float(score), "mafd": row.get("mafd"),
                "threshold": threshold, "min_scene_len_s": min_gap_s,
            })
    peaks = suppress_candidates(raw, min_gap_s)
    for index, row in enumerate(peaks, 1):
        row["boundary_id"] = f"B{index:04d}"
    write_csv(out_dir / "scene_b_ffmpeg_frames.csv", scores, ["frame_no", "time_s", "mafd", "score", "scd_time"])
    write_csv(out_dir / "scene_b_ffmpeg.csv", peaks,
              ["boundary_id", "frame_no", "time_s", "detector", "score", "mafd", "threshold", "min_scene_len_s"])
    write_json(out_dir / "scene_b_metadata.json", {
        "detector": "FFmpeg scdet", "ffmpeg_version": tool_version("ffmpeg"),
        "threshold": threshold, "min_scene_len_s": min_gap_s,
        "frame_score_count": len(scores), "candidate_count": len(peaks),
        "note": "scdet threshold uses a 0-100 scale and is not comparable to ContentDetector threshold numerically",
    })
    return peaks


def detect_transnetv2(video: Path, frames: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    """Optional adapter for the official TransNetV2 Python inference package.

    Put the official TransNetV2 inference module and its compatible dependencies on PYTHONPATH.
    This function intentionally fails clearly when unavailable instead of silently using another model.
    """
    try:
        from transnetv2 import TransNetV2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "TransNetV2 未安装/不可导入。请先按官方仓库说明安装依赖和权重，并将其推理模块加入 PYTHONPATH；"
            "本次没有自动降级，以免将其他检测器结果标记为 TransNetV2。"
        ) from exc
    model = TransNetV2()
    prediction = model.predict_video(str(video))
    if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
        raise RuntimeError("TransNetV2 predict_video 返回结构与当前适配器预期不一致；请按所安装版本更新 adapter。")
    single_frame_predictions = prediction[1]
    scenes = model.predictions_to_scenes(single_frame_predictions)
    candidates: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes[1:], 1):
        fn = int(scene[0])
        t = frame_pts(fn, frames)
        if t is None:
            continue
        candidates.append({"boundary_id": f"B{index:04d}", "frame_no": fn, "time_s": float(t),
                           "detector": "transnetv2", "score": None, "threshold": None,
                           "min_scene_len_s": None})
    write_csv(out_dir / "scene_b_transnetv2.csv", candidates,
              ["boundary_id", "frame_no", "time_s", "detector", "score", "threshold", "min_scene_len_s"])
    write_json(out_dir / "scene_b_metadata.json", {
        "detector": "TransNetV2", "candidate_count": len(candidates),
        "note": "Boundary candidates returned by the installed TransNetV2 package; verify API and weights version in environment.",
    })
    return candidates
