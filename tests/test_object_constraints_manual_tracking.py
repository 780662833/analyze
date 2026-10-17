"""Manual-init + LK tracking + preview tests (tasks 3 & 5 / M3-M4).

Covers: manual bbox seed recorded in initialization.json, per-frame box
updates derived from tracked point displacement (box follows the object),
feature-loss surfacing in quality rows, and a rendered review preview whose
frame count matches the window.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from omintel.object_constraints.analyze import (
    _seed_points_in_bbox,
    _tracked_box_from_points,
    run_frames,
)
from omintel.object_constraints.detector import BBox
from omintel.object_constraints.preview import render_preview_from_artifacts
from omintel.object_constraints.tracker import OpenCVLucasKanadeTracker
from l0_fixture import write_l0 as _write_l0


def _moving_block_frames(n: int = 12, size: int = 64, block: int = 16):
    """A textured block translating right; interior texture lets LK track."""
    rng = np.random.RandomState(7)
    texture = rng.randint(40, 220, (block, block)).astype(np.uint8)
    frames = []
    for i in range(n):
        img = np.full((size, size), 30, dtype=np.uint8)
        x = 8 + i * 3
        img[24:24 + block, x:x + block] = texture
        frames.append(img)
    return frames, block, 24


@unittest.skipUnless(HAS_CV2, "opencv-python not installed")
class BoxUpdateFromPointsTest(unittest.TestCase):
    def test_seed_points_strictly_inside_box(self):
        pts = _seed_points_in_bbox(BBox(0.25, 0.25, 0.75, 0.75), (100, 100))
        self.assertEqual(len(pts), 25)
        for x, y in pts:
            self.assertTrue(25 < x < 75 and 25 < y < 75)

    def test_box_follows_point_displacement(self):
        box = BBox(0.25, 0.25, 0.75, 0.75)
        points = [[[50 + 3 * f, 50.0] for f in range(6)] for _ in range(4)]
        vis = [[1.0] * 6 for _ in range(4)]
        boxes = _tracked_box_from_points(box, points, vis, (100, 100))
        self.assertAlmostEqual(boxes[0].x_min, 0.25, places=6)
        self.assertAlmostEqual(boxes[5].x_min, 0.25 + 15 / 100, places=6)
        # pure translation: size never changes
        for b in boxes:
            self.assertAlmostEqual(b.x_max - b.x_min, box.x_max - box.x_min, places=9)

    def test_box_holds_when_too_few_points_visible(self):
        box = BBox(0.2, 0.2, 0.6, 0.6)
        points = [[[30 + 10 * f, 40.0] for f in range(3)] for _ in range(4)]
        vis = [[1.0, 0.0, 0.0] for _ in range(4)]  # all lost after frame 0
        boxes = _tracked_box_from_points(box, points, vis, (100, 100))
        self.assertEqual(boxes[1], box)
        self.assertEqual(boxes[2], box)


@unittest.skipUnless(HAS_CV2, "opencv-python not installed")
class ManualInitTrackingLoopTest(unittest.TestCase):
    def test_manual_init_records_and_tracks_moving_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_facts, frames_csv = _write_l0(root, n_frames=12)
            frames, block, y0 = _moving_block_frames()
            out = root / "out"
            # normalized box around the block's initial position (+margin)
            seed = BBox((8 - 4) / 64, (24 - 4) / 64, (8 + block + 4) / 64, (24 + block + 4) / 64)

            summary = run_frames(
                frames,
                media_facts=media_facts,
                frames_csv=frames_csv,
                out_dir=out,
                object_id="obj_01",
                class_name="person",
                pts_start=0.0,
                pts_end=11 / 30.0,
                bbox=seed,
                init_frame_no=0,
                tracker=OpenCVLucasKanadeTracker(),
                track_box_updates=True,
                mask_source="manual_bbox_rect",
            )

            init = json.loads((out / "initialization.json").read_text(encoding="utf-8"))
            self.assertEqual(init["init_frame_no"], 0)
            self.assertEqual(init["source"], "manual_bbox")

            bboxes = np.load(out / "tracks" / "bbox.npz")["bboxes"]
            # the tracked box must follow the block to the right
            self.assertGreater(bboxes[-1][0], bboxes[0][0] + 0.2)

            # feature_loss_ratio column is populated by the real tracker
            with (out / "quality_per_frame.csv").open(encoding="utf-8-sig") as fh:
                import csv as _csv
                rows = list(_csv.DictReader(fh))
            self.assertTrue(all(r["feature_loss_ratio"] != "" for r in rows))

            # preview renders one frame per window frame, flags REVIEW frames
            from omintel.object_constraints.preview import load_quality_rows, render_preview

            result = render_preview(
                frames, list(range(12)), bboxes,
                load_quality_rows(out / "quality_per_frame.csv"),
                out / "preview.avi", fps=30.0, object_label="obj_01")
            self.assertEqual(result["frames_written"], 12)
            self.assertEqual(len(result["flagged_frames"]), 12)  # uncalibrated -> all REVIEW
            self.assertTrue(Path(result["preview_file"]).stat().st_size > 0)
            self.assertEqual(summary["gate"], "REVIEW")

    def test_init_frame_outside_window_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_facts, frames_csv = _write_l0(root, n_frames=6)
            frames = [np.zeros((64, 64), dtype=np.uint8) for _ in range(6)]
            with self.assertRaises(ValueError):
                run_frames(
                    frames, media_facts=media_facts, frames_csv=frames_csv,
                    out_dir=root / "out", object_id="obj_01",
                    pts_start=0.0, pts_end=5 / 30.0,
                    bbox=BBox(0.25, 0.25, 0.75, 0.75), init_frame_no=99,
                )


if __name__ == "__main__":
    unittest.main()
