"""Self-check for the sparse-flow tracker and per-frame quality assessment."""
from __future__ import annotations

import unittest

import numpy as np

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from omintel.object_constraints.quality import aggregate_gate, assess_frame
from omintel.object_constraints.tracker import OpenCVLucasKanadeTracker


def _moving_dot_frames(n: int = 10, h: int = 64, w: int = 64) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        img = np.zeros((h, w), dtype=np.uint8)
        x, y = 16 + i * 3, 20
        img[y - 1:y + 2, x - 1:x + 2] = 255
        frames.append(img)
    return frames


@unittest.skipUnless(HAS_CV2, "opencv-python not installed")
class LucasKanadeTrackerTest(unittest.TestCase):
    def test_tracks_a_moving_dot(self):
        frames = _moving_dot_frames()
        tracker = OpenCVLucasKanadeTracker()
        result = tracker.track(frames, [[16.0, 20.0]], (0, len(frames) - 1))

        pts = result["points"][0]
        self.assertEqual(len(pts), len(frames))
        self.assertAlmostEqual(pts[0][0], 16.0, delta=1.0)
        self.assertAlmostEqual(pts[0][1], 20.0, delta=1.0)
        # motion is tracked to the right across the sequence
        self.assertGreater(pts[-1][0], pts[0][0])
        self.assertTrue(all(v == 1.0 for v in result["visibility"][0]))

    def test_rejects_invalid_time_range(self):
        tracker = OpenCVLucasKanadeTracker()
        with self.assertRaises(ValueError):
            tracker.track(_moving_dot_frames(3), [[16.0, 20.0]], (0, 5))


class QualityAssessmentTest(unittest.TestCase):
    def test_uncalibrated_defaults(self):
        row = assess_frame(frame_no=210, pts_time_s=7.0, mask_area_ratio_change=0.02)
        self.assertIsNone(row["anomaly_score"])
        self.assertEqual(row["gate"], "REVIEW")

    def test_aggregate_gate_ordering(self):
        self.assertEqual(aggregate_gate([{"gate": "REVIEW"}, {"gate": "PASS"}]), "REVIEW")
        self.assertEqual(aggregate_gate([{"gate": "REPAIR"}, {"gate": "PASS"}]), "REPAIR")
        self.assertEqual(aggregate_gate([{"gate": "PASS"}]), "PASS")
        self.assertEqual(aggregate_gate([]), "REVIEW")


if __name__ == "__main__":
    unittest.main()