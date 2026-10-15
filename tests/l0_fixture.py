"""Shared L0 fixture writer for object-constraint tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def write_l0(root: Path, n_frames: int = 10, size: int = 64, fps: float = 30.0):
    media_facts = {
        "source": {"sha256": "a" * 64},
        "video_stream": {
            "width": size, "height": size,
            "display_width": size, "display_height": size, "rotation_deg": 0,
        },
        "frame_timeline": {"likely_variable_frame_rate": False},
    }
    (root / "media_facts.json").write_text(json.dumps(media_facts), encoding="utf-8")
    frames_path = root / "frames.csv"
    with frames_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_no", "pts_time_s"])
        for i in range(n_frames):
            writer.writerow([i, i / fps])
    return root / "media_facts.json", frames_path
