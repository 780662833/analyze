"""Self-check for the object-level constraint data contracts.

Run with the repo's unittest convention:
``python -m unittest discover -s tests -v``

The sample values below are illustrative. PTS values come from the canonical
``frames.csv`` timeline and must not be derived from ``frame_no / fps``.
"""
from __future__ import annotations

import unittest

from omintel.object_constraints.schemas import (
    build_object_constraints,
    build_object_tracks,
    validate_object_constraints,
    validate_object_tracks,
)


def _sample_tracks() -> dict:
    return build_object_tracks(
        video_id="3f2a9c1d8b7e6a55",
        source_sha256="a" * 64,
        object_id="shoe_01",
        shot_id="S0004",
        class_name="sneaker",
        coded_size=(1920, 1080),
        display_size=(1920, 1080),
        rotation_deg=0,
        start_frame_no=210, start_pts_s=7.0,
        end_frame_no=314, end_pts_s=10.5,
        likely_variable_frame_rate=False,
        bbox_file="tracks/bbox.npz",
        mask_dir="masks",
        mask_index_csv="masks/mask_index.csv",
        motion_file="tracks/motion_track.npz",
        per_frame_quality_csv="quality_per_frame.csv",
    )


def _sample_constraints() -> dict:
    return build_object_constraints(
        video_id="3f2a9c1d8b7e6a55",
        source_sha256="a" * 64,
        object_id="shoe_01",
        coded_size=(1920, 1080),
        display_size=(1920, 1080),
        rotation_deg=0,
        start_frame_no=210, start_pts_s=7.0,
        end_frame_no=314, end_pts_s=10.5,
    )


class ObjectContractTests(unittest.TestCase):
    def test_tracks_contract_is_valid(self):
        validate_object_tracks(_sample_tracks())

    def test_constraints_contract_is_valid(self):
        validate_object_constraints(_sample_constraints())

    def test_tracks_rejects_inlined_data(self):
        doc = _sample_tracks()
        doc["tracks"]["mask_array"] = [[0, 1], [1, 0]]
        with self.assertRaises(ValueError):
            validate_object_tracks(doc)

    def test_constraints_defaults_to_review_when_uncalibrated(self):
        doc = _sample_constraints()
        self.assertEqual(doc["quality_gate"]["status"], "REVIEW")
        self.assertTrue(all(sc["weight"] is None for sc in doc["soft_constraints"]))


if __name__ == "__main__":
    unittest.main()