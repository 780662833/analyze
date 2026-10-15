"""Cross-artifact consistency audit (task 1 / M1).

After a closed-loop run, every sidecar must describe the *same* frames in the
*same* coordinate space as the JSON contracts. These tests read the written
artifacts back from disk and check them against each other -- not just against
the builder inputs -- so a writer bug cannot hide behind a matching builder:

- object_tracks.json time_base window == mask_index.csv rows == quality CSV
  rows == bbox.npz rows == motion_track.npz frame axis;
- mask_index.csv PTS == frames.csv PTS for the same frame_no;
- mask PNG pixel size == media_reference coded size;
- bbox values stay inside the declared normalized xyxy space;
- initialization.json init_frame_no lies inside the window;
- tracks.mask_source is declared (rect placeholder vs real segmentation).
"""
from __future__ import annotations

import csv
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

from omintel.object_constraints.analyze import run_frames
from omintel.object_constraints.detector import BBox
from l0_fixture import write_l0 as _write_l0


def _run_loop(root: Path, n_frames: int = 8):
    media_facts, frames_csv = _write_l0(root, n_frames=n_frames)
    frames = [np.zeros((64, 64), dtype=np.uint8) for _ in range(n_frames)]
    out = root / "out"
    summary = run_frames(
        frames,
        media_facts=media_facts,
        frames_csv=frames_csv,
        out_dir=out,
        object_id="obj_01",
        class_name="person",
        pts_start=0.0,
        pts_end=(n_frames - 1) / 30.0,
        bbox=BBox(0.25, 0.25, 0.75, 0.75),
        init_frame_no=0,
        mask_source="manual_bbox_rect",
    )
    return summary, out, frames_csv


@unittest.skipUnless(HAS_CV2, "opencv-python not installed")
class ArtifactConsistencyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.summary, self.out, self.frames_csv = _run_loop(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _tracks(self):
        return json.loads((self.out / "object_tracks.json").read_text(encoding="utf-8"))

    def test_all_artifacts_cover_the_same_frames(self):
        tracks = self._tracks()
        tb = tracks["time_base"]
        start, end = tb["start"]["frame_no"], tb["end"]["frame_no"]
        expected = list(range(start, end + 1))

        with (self.out / "masks" / "mask_index.csv").open(encoding="utf-8-sig") as fh:
            mask_rows = list(csv.DictReader(fh))
        self.assertEqual([int(r["frame_no"]) for r in mask_rows], expected)

        with (self.out / "quality_per_frame.csv").open(encoding="utf-8-sig") as fh:
            q_rows = list(csv.DictReader(fh))
        self.assertEqual([int(r["frame_no"]) for r in q_rows], expected)

        bboxes = np.load(self.out / "tracks" / "bbox.npz")["bboxes"]
        self.assertEqual(len(bboxes), len(expected))

        motion = np.load(self.out / "tracks" / "motion_track.npz")
        self.assertEqual(motion["points"].shape[1], len(expected))
        self.assertEqual(motion["visibility"].shape[1], len(expected))

    def test_mask_index_pts_matches_frames_csv(self):
        with self.frames_csv.open(encoding="utf-8-sig") as fh:
            by_no = {int(r["frame_no"]): r["pts_time_s"] for r in csv.DictReader(fh)}
        with (self.out / "masks" / "mask_index.csv").open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                self.assertAlmostEqual(float(r["pts_time_s"]), float(by_no[int(r["frame_no"])]),
                                       places=9, msg=f"PTS drift at frame {r['frame_no']}")

    def test_mask_png_size_matches_coded_size(self):
        tracks = self._tracks()
        w = tracks["media_reference"]["coded_width"]
        h = tracks["media_reference"]["coded_height"]
        with (self.out / "masks" / "mask_index.csv").open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                img = cv2.imread(str(self.out / r["mask_file"]))
                self.assertIsNotNone(img, f"unreadable mask {r['mask_file']}")
                self.assertEqual((img.shape[1], img.shape[0]), (w, h))

    def test_bboxes_within_declared_coordinate_space(self):
        bboxes = np.load(self.out / "tracks" / "bbox.npz")["bboxes"]
        self.assertTrue(np.all(bboxes >= 0.0) and np.all(bboxes <= 1.0))
        self.assertTrue(np.all(bboxes[:, 2] > bboxes[:, 0]))
        self.assertTrue(np.all(bboxes[:, 3] > bboxes[:, 1]))

    def test_initialization_sidecar_is_consistent_with_window(self):
        tracks = self._tracks()
        self.assertEqual(tracks["tracks"]["initialization_file"], "initialization.json")
        init = json.loads((self.out / "initialization.json").read_text(encoding="utf-8"))
        tb = tracks["time_base"]
        self.assertTrue(tb["start"]["frame_no"] <= init["init_frame_no"] <= tb["end"]["frame_no"])
        self.assertEqual(init["object_id"], "obj_01")
        self.assertEqual(init["source"], "manual_bbox")
        self.assertEqual(init["bbox_xyxy_normalized"], [0.25, 0.25, 0.75, 0.75])

    def test_mask_source_declares_placeholder_precision(self):
        tracks = self._tracks()
        self.assertEqual(tracks["tracks"]["mask_source"], "manual_bbox_rect")
        # a rect-derived mask must never be advertised as an object outline
        self.assertEqual(tracks["tracks"]["mask_precision"], "rectangular_approximation")

    def test_quality_gate_reason_column_present_and_populated(self):
        with (self.out / "quality_per_frame.csv").open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(all("gate_reason" in r and r["gate"] in ("PASS", "REPAIR", "REVIEW")
                            for r in rows))


if __name__ == "__main__":
    unittest.main()
