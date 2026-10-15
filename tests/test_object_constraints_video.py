"""Real-video decode tests: cv2.VideoCapture frame indexing + drop detection."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from omintel.object_constraints.video_io import decode_window, video_info


def _make_synthetic_avi(path: Path, n_frames: int = 15, size: int = 48) -> None:
    # MJPG is intra-frame only, so seeking lands exactly on the target frame,
    # which is what the decode_window indexing contract assumes.
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30.0,
        (size, size),
    )
    for i in range(n_frames):
        frame = np.zeros((size, size), dtype=np.uint8)
        frame[i % size, i % size] = 255
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


@unittest.skipUnless(HAS_CV2, "opencv-python not installed")
class VideoDecodeTest(unittest.TestCase):
    def test_decode_window_returns_exact_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            avi = Path(tmp) / "synthetic.avi"
            _make_synthetic_avi(avi, n_frames=15)

            info = video_info(avi)
            self.assertEqual(info["frame_count"], 15)

            frames = decode_window(avi, 2, 6)  # 5 frames: 2..6 inclusive
            self.assertEqual(len(frames), 5)
            self.assertEqual(frames[0].shape, (48, 48, 3))

    def test_decode_window_raises_on_early_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            avi = Path(tmp) / "synthetic.avi"
            _make_synthetic_avi(avi, n_frames=5)

            with self.assertRaises(RuntimeError):
                decode_window(avi, 0, 9)  # asks for 10, only 5 exist


if __name__ == "__main__":
    unittest.main()