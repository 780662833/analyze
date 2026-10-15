from __future__ import annotations
import bisect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import sha256_file, write_csv, write_json


def fuse_candidates(a: list[dict[str, Any]], b: list[dict[str, Any]], tolerance_s: float) -> list[dict[str, Any]]:
    """Optimal monotonic one-to-one matching: maximize matches, then minimize total time delta.

    Agreement is corroboration only, never a claim that the boundary is correct.
    """
    aa = sorted(a, key=lambda x: float(x["time_s"]))
    bb = sorted(b, key=lambda x: float(x["time_s"]))
    n, m = len(aa), len(bb)
    # Dynamic programming over the sorted timelines. The objective is lexicographic:
    # (1) maximum number of pairs within tolerance, (2) minimum total absolute delta.
    matches = [[0] * (m + 1) for _ in range(n + 1)]
    costs = [[0.0] * (m + 1) for _ in range(n + 1)]
    actions = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        actions[i][0] = "skip_a"
    for j in range(1, m + 1):
        actions[0][j] = "skip_b"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Start with skipping the A candidate.
            best_count = matches[i - 1][j]
            best_cost = costs[i - 1][j]
            best_action = "skip_a"
            # Skipping B can be better, or a deterministic tie-break.
            count_b, cost_b = matches[i][j - 1], costs[i][j - 1]
            if count_b > best_count or (count_b == best_count and cost_b < best_cost - 1e-12):
                best_count, best_cost, best_action = count_b, cost_b, "skip_b"
            delta = abs(float(aa[i - 1]["time_s"]) - float(bb[j - 1]["time_s"]))
            if delta <= tolerance_s:
                count_m = matches[i - 1][j - 1] + 1
                cost_m = costs[i - 1][j - 1] + delta
                if count_m > best_count or (count_m == best_count and cost_m <= best_cost + 1e-12):
                    best_count, best_cost, best_action = count_m, cost_m, "match"
            matches[i][j], costs[i][j], actions[i][j] = best_count, best_cost, best_action

    paired: list[tuple[int, int, float]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    i, j = n, m
    while i > 0 or j > 0:
        action = actions[i][j] if i >= 0 and j >= 0 else ""
        if i > 0 and j > 0 and action == "match":
            delta = abs(float(aa[i - 1]["time_s"]) - float(bb[j - 1]["time_s"]))
            paired.append((i - 1, j - 1, delta))
            used_a.add(i - 1); used_b.add(j - 1)
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or action == "skip_a"):
            i -= 1
        elif j > 0:
            j -= 1
        else:
            break
    fused: list[dict[str, Any]] = []
    for ai, bi, delta in paired:
        ta, tb = float(aa[ai]["time_s"]), float(bb[bi]["time_s"])
        fused.append({"time_s": (ta + tb) / 2.0, "status": "agreement", "support_count": 2,
                      "delta_s": delta, "scene_a": aa[ai], "scene_b": bb[bi]})
    for ai, row in enumerate(aa):
        if ai not in used_a:
            fused.append({"time_s": float(row["time_s"]), "status": "scene_a_only", "support_count": 1,
                          "delta_s": None, "scene_a": row, "scene_b": None})
    for bi, row in enumerate(bb):
        if bi not in used_b:
            fused.append({"time_s": float(row["time_s"]), "status": "scene_b_only", "support_count": 1,
                          "delta_s": None, "scene_a": None, "scene_b": row})
    fused.sort(key=lambda x: float(x["time_s"]))
    for index, row in enumerate(fused, 1):
        row["candidate_id"] = f"C{index:04d}"
        row["time_s"] = round(float(row["time_s"]), 9)
        if row["delta_s"] is not None:
            row["delta_s"] = round(float(row["delta_s"]), 9)
    return fused


def write_timeline_outputs(out_dir: Path, fused: list[dict[str, Any]], facts: dict[str, Any]) -> list[dict[str, Any]]:
    duration = facts.get("container", {}).get("duration_s")
    flat = []
    for row in fused:
        a, b = row.get("scene_a"), row.get("scene_b")
        flat.append({
            "candidate_id": row["candidate_id"], "time_s": row["time_s"], "status": row["status"],
            "support_count": row["support_count"], "delta_s": row["delta_s"],
            "scene_a_detector": (a or {}).get("detector"), "scene_a_score": (a or {}).get("score"),
            "scene_b_detector": (b or {}).get("detector"), "scene_b_score": (b or {}).get("score"),
            "scene_a_frame_no": (a or {}).get("frame_no"), "scene_b_frame_no": (b or {}).get("frame_no"),
        })
    write_csv(out_dir / "shot_candidates.csv", flat,
              ["candidate_id", "time_s", "status", "support_count", "delta_s", "scene_a_detector", "scene_a_score", "scene_b_detector", "scene_b_score", "scene_a_frame_no", "scene_b_frame_no"])
    source = facts["source"]
    write_json(out_dir / "shot_candidates.json", {
        "schema_version": "openmontage.shot_candidates.v1",
        "video_id": source["sha256"][:16], "source_sha256": source["sha256"],
        "duration_s": duration, "boundaries": flat,
        "interpretation_note": "Candidates only. agreement means two detector candidates occur within configured tolerance; it is not proof of ground-truth correctness.",
    })
    shots: list[dict[str, Any]] = []
    if duration is not None and float(duration) > 0:
        timeline = [(0.0, "start", "video_start")]
        for row in fused:
            t = float(row["time_s"])
            if 0.001 < t < float(duration) - 0.001:
                timeline.append((t, row["status"], row["candidate_id"]))
        timeline.append((float(duration), "end", "video_end"))
        timeline.sort(key=lambda x: x[0])
        dedup: list[tuple[float, str, str]] = []
        for item in timeline:
            if dedup and abs(item[0] - dedup[-1][0]) < 1e-6:
                if item[1] == "agreement":
                    dedup[-1] = item
            else:
                dedup.append(item)
        for idx in range(len(dedup) - 1):
            start, start_status, start_id = dedup[idx]
            end, end_status, end_id = dedup[idx + 1]
            if end <= start:
                continue
            shots.append({"shot_id": f"S{idx+1:04d}", "start_s": round(start, 9), "end_s": round(end, 9),
                          "duration_s": round(end - start, 9), "start_boundary_support": start_status,
                          "end_boundary_support": end_status, "start_boundary_id": start_id,
                          "end_boundary_id": end_id, "provisional": True})
    write_csv(out_dir / "provisional_shots.csv", shots,
              ["shot_id", "start_s", "end_s", "duration_s", "start_boundary_support", "end_boundary_support", "start_boundary_id", "end_boundary_id", "provisional"])
    return flat


def nearest_frame_index(t: float, frames: list[dict[str, Any]]) -> int | None:
    valid = [(float(r["pts_time_s"]), int(r["frame_no"])) for r in frames if r.get("pts_time_s") is not None]
    if not valid:
        return None
    times = [x[0] for x in valid]
    # Use bisection for monotonic PTS, otherwise fall back to global nearest search.
    monotonic = all(x <= y for x, y in zip(times, times[1:]))
    if not monotonic:
        return min(valid, key=lambda x: abs(x[0] - t))[1]
    pos = bisect.bisect_left(times, t)
    options = []
    if pos < len(valid): options.append(valid[pos])
    if pos > 0: options.append(valid[pos - 1])
    return min(options, key=lambda x: abs(x[0] - t))[1]


def build_evidence_pack(video: Path, out_dir: Path, fused: list[dict[str, Any]],
                        frames: list[dict[str, Any]], facts: dict[str, Any]) -> dict[str, Any]:
    """L2 evidence pack: for each boundary event, save before/at/after frames and hashes."""
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("事件证据帧提取需要 OpenCV：python -m pip install opencv-python") from exc

    source_path = Path(video).resolve()
    evidence_root = out_dir / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    targets: dict[int, list[tuple[str, str, float | None]]] = {}
    event_records: list[dict[str, Any]] = []

    for row in fused:
        center_idx = nearest_frame_index(float(row["time_s"]), frames)
        if center_idx is None:
            event_records.append({"event_id": row["candidate_id"], "event_type": "shot_boundary_candidate",
                                  "time_s": row["time_s"], "status": row["status"], "evidence_status": "no_valid_pts"})
            continue
        idx_roles = {"before": max(0, center_idx - 1), "at": center_idx, "after": min(len(frames)-1, center_idx + 1)}
        frame_refs = {}
        for role, fn in idx_roles.items():
            ts = frames[fn].get("pts_time_s") if 0 <= fn < len(frames) else None
            targets.setdefault(fn, []).append((row["candidate_id"], role, ts))
            frame_refs[role] = {"frame_no": fn, "pts_time_s": ts,
                                "path": f"evidence/{row['candidate_id']}/{role}.jpg"}
        event_records.append({
            "event_id": row["candidate_id"], "event_type": "shot_boundary_candidate",
            "time_s": row["time_s"], "status": row["status"], "support_count": row["support_count"],
            "delta_s": row["delta_s"],
            "sources": [s for s in (row.get("scene_a"), row.get("scene_b")) if s is not None],
            "evidence_status": "pending_extraction", "frames": frame_refs,
        })

    saved: dict[tuple[str, str], dict[str, Any]] = {}
    if targets:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频用于证据抽帧：{source_path}")
        max_target = max(targets)
        fn = 0
        while fn <= max_target:
            ok, image = cap.read()
            if not ok:
                break
            if fn in targets:
                for event_id, role, ts in targets[fn]:
                    target_path = evidence_root / event_id / f"{role}.jpg"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                        continue
                    saved[(event_id, role)] = {
                        "frame_no": fn, "pts_time_s": ts,
                        "path": str(target_path.relative_to(out_dir)).replace("\\", "/"),
                        "sha256": sha256_file(target_path), "width": int(image.shape[1]), "height": int(image.shape[0]),
                    }
            fn += 1
        cap.release()

    for event in event_records:
        if "frames" not in event:
            continue
        extracted = {}
        for role, original in event["frames"].items():
            detail = saved.get((event["event_id"], role))
            extracted[role] = detail or {**original, "status": "frame_decode_failed"}
        event["frames"] = extracted
        event["evidence_status"] = "complete" if all("sha256" in r for r in extracted.values()) else "partial"

    package = {
        "schema_version": "openmontage.evidence_pack.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_id": facts["source"]["sha256"][:16], "source_sha256": facts["source"]["sha256"],
        "event_count": len(event_records), "events": event_records,
        "note": "Shot-boundary candidates and neighboring frames only. This does not yet label action semantics or prove cut correctness.",
    }
    write_json(out_dir / "events.json", package)
    write_json(out_dir / "evidence_index.json", {
        "schema_version": "openmontage.evidence_index.v1", "video_id": package["video_id"],
        "source_sha256": package["source_sha256"],
        "frames": [{"event_id": ev["event_id"], "role": role, **frame}
                   for ev in event_records for role, frame in (ev.get("frames") or {}).items()],
    })
    return package
