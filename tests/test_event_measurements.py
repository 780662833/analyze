"""L1.5 Commit E tests: deterministic luminance measurement sequence (E1 core).

Covers the synthetic cases mandated for brightness measurement:
constant, step, gradual ramp -- plus PTS-authority, explicit gaps, and
byte-level determinism.  No LLM / no GPU / no network.
"""
from __future__ import annotations

import unittest

import numpy as np

from omintel.event_measurements import (
    ALGORITHM_VERSION,
    check_measurement_determinism,
    compute_luminance_sequence,
)


def _frames(n: int, pts_step: float = 0.037):
    """Deliberately non-integer, non-fps-derived PTS values."""
    return [
        {"frame_no": i, "pts_time_s": f"{i * pts_step:.6f}"}
        for i in range(n)
    ]


def _gray(value: int, size: int = 8) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


class LuminanceSequenceTests(unittest.TestCase):
    def test_constant_brightness_zero_deltas(self):
        frames = _frames(6)
        images = [_gray(120) for _ in frames]
        rec = compute_luminance_sequence(frames, images)
        self.assertEqual(rec["delta_mean_luma"], [0.0] * 5)
        self.assertEqual(rec["mean_luma"], [120.0] * 6)
        self.assertEqual(rec["invalid_frames"], [])
        self.assertEqual(rec["gaps"], [])
        self.assertEqual(rec["coverage_ratio"], 1.0)

    def test_step_change_recorded_with_exact_pts(self):
        frames = _frames(6)
        vals = [100, 100, 100, 150, 150, 150]
        images = [_gray(v) for v in vals]
        rec = compute_luminance_sequence(frames, images)
        self.assertEqual(rec["delta_mean_luma"], [0.0, 0.0, 50.0, 0.0, 0.0])
        # step lands between frame 2 and 3 -> PTS pair consumed verbatim
        self.assertAlmostEqual(rec["pts_s"][3], 3 * 0.037, places=6)
        self.assertNotEqual(rec["pts_s"][3], 3 / 30.0)  # never frame_no/fps
        self.assertEqual(rec["algorithm_version"], ALGORITHM_VERSION)

    def test_gradual_ramp_small_uniform_deltas(self):
        frames = _frames(8)
        images = [_gray(50 + i) for i in range(8)]
        rec = compute_luminance_sequence(frames, images)
        self.assertTrue(all(d == 1.0 for d in rec["delta_mean_luma"]))

    def test_missing_pts_becomes_explicit_gap_not_drop(self):
        frames = _frames(4)
        frames[2]["pts_time_s"] = ""          # unparseable
        frames[3]["pts_time_s"] = None        # null
        images = [_gray(v) for v in (10, 20, 30, 40)]
        rec = compute_luminance_sequence(frames, images)
        self.assertEqual(len(rec["frame_nos"]), 4)   # nothing dropped
        gap_reasons = {g["reason"] for g in rec["gaps"]}
        self.assertIn("missing_pts", gap_reasons)
        gap_frames = {g["frame_range"]["start"] for g in rec["gaps"]
                      if g["reason"] == "missing_pts"}
        self.assertEqual(gap_frames, {2, 3})
        self.assertIsNone(rec["pts_s"][2])
        self.assertIsNone(rec["pts_s"][3])
        # luma still measured where image exists; deltas unaffected by PTS gap
        self.assertEqual(rec["delta_mean_luma"], [10.0, 10.0, 10.0])

    def test_unreadable_frame_gapped_not_silently_ignored(self):
        frames = _frames(5)
        images = [_gray(10), _gray(10), None, _gray(80), _gray(80)]
        rec = compute_luminance_sequence(frames, images)
        self.assertEqual(rec["invalid_frames"], [2])
        self.assertIsNone(rec["mean_luma"][2])
        self.assertIsNone(rec["delta_mean_luma"][1])
        self.assertIsNone(rec["delta_mean_luma"][2])
        det = [g["detail"] for g in rec["gaps"]]
        self.assertTrue(any("unreadable" in d for d in det))
        self.assertLess(rec["coverage_ratio"], 1.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_luminance_sequence(_frames(3), [_gray(1)])

    def test_deterministic_identical_inputs_identical_record(self):
        frames = _frames(10, pts_step=1 / 29.97)
        rng = np.random.default_rng(0)
        images = [(rng.integers(0, 256, (8, 8))).astype(np.uint8)
                  for _ in frames]
        a = compute_luminance_sequence(frames, images)
        b = compute_luminance_sequence(frames, images)
        self.assertEqual(repr(a), repr(b))
        check_measurement_determinism(a)

    def test_params_recorded_in_provenance(self):
        rec = compute_luminance_sequence(_frames(2), [_gray(5), _gray(9)],
                                         params={"stat_precision": 3})
        self.assertEqual(rec["params"]["stat_precision"], 3)
        self.assertEqual(rec["algorithm"], "luminance_sequence")


if __name__ == "__main__":
    unittest.main()
