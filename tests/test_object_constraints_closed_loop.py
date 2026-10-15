"""Model-free closed-loop test: frames -> per-frame data -> sidecars + contracts."""
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
from omintel.object_constraints.schemas import (
    validate_object_constraints,
    validate_object_tracks,
)


import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from l0_fixture import write_l0 as _write_l0  # noqa: E402


@unittest.skipUnless(HAS_CV2, "opencv-python not installed")
class ClosedLoopTest(unittest.TestCase):
    def test_full_loop_writes_sidecars_and_contracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_facts, frames_csv = _write_l0(root, n_frames=10)

            frames = [np.zeros((64, 64), dtype=np.uint8) for _ in range(10)]
            out = root / "out"

            summary = run_frames(
                frames,
                media_facts=media_facts,
                frames_csv=frames_csv,
                out_dir=out,
                object_id="obj_01",
                class_name="person",
                pts_start=0.0,
                pts_end=9 / 30.0,
                bbox=BBox(0.25, 0.25, 0.75, 0.75),
            )

            tracks = json.loads((out / "object_tracks.json").read_text(encoding="utf-8"))
            constraints = json.loads((out / "object_constraints.json").read_text(encoding="utf-8"))
            validate_object_tracks(tracks)
            validate_object_constraints(constraints)

            masks = sorted((out / "masks").glob("frame_*.png"))
            self.assertEqual(len(masks), 10)
            self.assertTrue((out / "masks" / "mask_index.csv").exists())
            self.assertTrue((out / "tracks" / "bbox.npz").exists())
            self.assertTrue((out / "tracks" / "motion_track.npz").exists())
            self.assertTrue((out / "quality_per_frame.csv").exists())

            self.assertEqual(summary["gate"], "REVIEW")
            self.assertEqual(tracks["tracks"]["mask_index_csv"], "masks/mask_index.csv")
            # index source fields derive from the same adapter backends as
            # provenance, so the two blocks can never drift apart
            self.assertEqual(tracks["tracks"]["detection_source"], "StaticBoxDetector")
            self.assertEqual(tracks["tracks"]["segmentation_source"], "BoxMaskSegmenter")
            self.assertEqual(tracks["tracks"]["motion_source"], "NullPointTracker")
            # provenance reflects the stub backends, not the real-model defaults
            self.assertIn("StaticBoxDetector", tracks["provenance"]["detector"]["backend"])
            # coordinate spaces are declared per sidecar so bbox/mask/motion
            # conventions cannot be misread (A3)
            self.assertEqual(tracks["tracks"]["bbox_coordinate_space"], "normalized_relative_to_coded_pixels")
            self.assertEqual(tracks["tracks"]["mask_coordinate_space"], "coded_pixels_pre_rotation")
            self.assertEqual(tracks["tracks"]["motion_coordinate_space"], "coded_pixels_pre_rotation")
            # the per-frame gate records why it is REVIEW (A4)
            quality_header = (out / "quality_per_frame.csv").read_text(encoding="utf-8-sig").splitlines()[0]
            self.assertIn("gate_reason", quality_header)


if __name__ == "__main__":
    unittest.main()