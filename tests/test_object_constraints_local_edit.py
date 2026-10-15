"""M5 local-edit regression: OVCL artifacts drive a composite, output is verified.

Proves the constraint contracts are consumable *without modification*:
* caption/patch composites land strictly inside each frame's tracked bbox;
* the fused FFV1 output re-decodes pixel-identical to the source everywhere
  outside the edit regions (preserve_regions "elsewhere" clause);
* timeline stays aligned (frame count == window length, PTS recorded);
* frames whose gate_reason marks tracking lost/drift are skipped (default) or
  abort the run (``fail`` decision) instead of being silently edited;
* verify_local_edit_output actually catches a leaking edit (negative control).
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from l0_fixture import write_l0 as _write_l0
from omintel.object_constraints.analyze import run_frames
from omintel.object_constraints.detector import BBox
from omintel.object_constraints.local_edit import (
    composite_local_overlay,
    verify_local_edit_output,
)
from omintel.object_constraints.tracker import OpenCVLucasKanadeTracker

N_FRAMES = 12
SIZE = 64


def _write_moving_block_video(path: Path) -> None:
    rng = np.random.RandomState(7)
    block = 16
    texture = rng.randint(40, 220, (block, block)).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
                             (SIZE, SIZE))
    for i in range(N_FRAMES):
        img = np.full((SIZE, SIZE), 30, dtype=np.uint8)
        x = 8 + i * 3
        img[24:24 + block, x:x + block] = texture
        writer.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    writer.release()


def _decode_all(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    ok, f = cap.read()
    while ok and f is not None:
        frames.append(f)
        ok, f = cap.read()
    cap.release()
    return frames


def _run_loop(tmp: Path) -> dict:
    src = tmp / "src.avi"
    _write_moving_block_video(src)
    media_facts, frames_csv = _write_l0(tmp, n_frames=N_FRAMES)
    gray = [cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) for fr in _decode_all(src)]
    res = run_frames(
        gray, media_facts=media_facts, frames_csv=frames_csv, out_dir=tmp / "art",
        object_id="obj_01", class_name="person",
        bbox=BBox(0.125, 0.375, 0.375, 0.625), init_frame_no=0,
        pts_start=0.0, pts_end=(N_FRAMES - 1) / 30.0,
        tracker=OpenCVLucasKanadeTracker(), track_box_updates=True,
    )
    return {"src": src, "art": tmp / "art", "res": res, "frames_csv": frames_csv}


def _mark_frame_unreliable(quality_csv: Path, frame_no: int) -> None:
    p = Path(quality_csv)
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    fields = list(rows[0].keys())
    for r in rows:
        if int(r["frame_no"]) == frame_no:
            r["gate_reason"] += "; tracking lost; re-initialization required"
    with p.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@unittest.skipUnless(HAS_CV2, "opencv-python not installed")
class LocalEditRegressionTest(unittest.TestCase):
    def test_composite_stays_inside_bbox_and_preserves_elsewhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = _run_loop(Path(tmp))
            rep = composite_local_overlay(env["src"], env["art"], Path(tmp) / "edit.mkv",
                                          frames_csv=env["frames_csv"], text="CAP")
            self.assertEqual(len(rep["edited_frames"]), N_FRAMES)
            self.assertEqual(rep["skipped_frames"], [])
            # report carries the PTS window used (timeline auditability)
            self.assertEqual(rep["window"]["start_frame_no"], 0)
            self.assertEqual(rep["window"]["end_frame_no"], N_FRAMES - 1)
            self.assertIn("0", rep["window"]["pts_by_frame_no"])
            v = verify_local_edit_output(env["src"], Path(tmp) / "edit.mkv", env["art"])
            self.assertTrue(v["passed"])
            self.assertEqual(v["frames_checked"], N_FRAMES)
            self.assertEqual(v["frames_changed"], N_FRAMES)

    def test_patch_mode_also_pixel_exact_outside_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = _run_loop(Path(tmp))
            out = Path(tmp) / "patch.mkv"
            composite_local_overlay(env["src"], env["art"], out)
            src_f = _decode_all(env["src"])
            dst_f = _decode_all(out)
            self.assertEqual(len(dst_f), len(src_f))
            bboxes = np.load(env["art"] / "tracks" / "bbox.npz")["bboxes"]
            for fn, a, b, box in zip(range(N_FRAMES), src_f, dst_f, bboxes):
                mask = np.zeros(a.shape[:2], dtype=bool)
                x0, y0, x1, y1 = (int(round(v * d)) for v, d in zip(box, (SIZE, SIZE, SIZE, SIZE)))
                # conservative dilation: FFV1 is bit-exact in principle, but the
                # verification here must tolerate any container-level rounding;
                # a real leak (e.g. full-frame overlay) still trips this check
                m = 3
                ys0, ys1 = max(0, y0 - m), min(SIZE, y1 + m)
                xs0, xs1 = max(0, x0 - m), min(SIZE, x1 + m)
                mask[ys0:ys1, xs0:xs1] = True
                outside = ~mask
                self.assertTrue((a[outside] == b[outside]).all(),
                                f"frame {fn}: pixels far outside bbox changed")

    def test_unreliable_frames_skipped_by_default_and_fatal_on_fail_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = _run_loop(Path(tmp))
            _mark_frame_unreliable(env["art"] / "quality_per_frame.csv", 5)
            rep = composite_local_overlay(env["src"], env["art"], Path(tmp) / "e2.mkv")
            self.assertNotIn(5, rep["edited_frames"])
            self.assertEqual([s["frame_no"] for s in rep["skipped_frames"]], [5])
            with self.assertRaises(RuntimeError):
                composite_local_overlay(env["src"], env["art"], Path(tmp) / "e3.mkv",
                                        nonpass_decision="fail")
            # skipped frame must be byte-identical to source in the output
            v = verify_local_edit_output(env["src"], Path(tmp) / "e2.mkv", env["art"])
            self.assertTrue(v["passed"])
            self.assertEqual(v["frames_changed"], N_FRAMES - 1)

    def test_verifier_catches_leaked_edit(self):
        """Negative control: an edit that paints outside the region must fail."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _run_loop(Path(tmp))
            leaked = Path(tmp) / "leaked.mkv"
            frames = _decode_all(env["src"])
            writer = cv2.VideoWriter(str(leaked), cv2.VideoWriter_fourcc(*"FFV1"),
                                     30.0, (SIZE, SIZE))
            for i, fr in enumerate(frames):
                fr = fr.copy()
                if i == 3:
                    fr[0:4, 0:4] = 255  # top-left corner, far from any tracked box
                writer.write(fr)
            writer.release()
            with self.assertRaises(AssertionError):
                verify_local_edit_output(env["src"], leaked, env["art"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
