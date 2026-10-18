"""L1.5 Commit E (E1b) tests: brightness_jump candidate criteria.

Mandated acceptance scenarios from the reviewed design:
  * single obvious step          -> one candidate on the correct frame pair,
                                    PTS taken from the authoritative record
  * tiny encoding noise          -> no candidate
  * linear ramp                  -> no per-step spam (cumulative change is
                                    NOT a sequence of jumps)
  * consecutive large deltas     -> deterministic merge into ONE event
  * confirmed hard-cut neighbor  -> suppressed as intra-shot claim, kept in
                                    boundary_attributed_transitions w/ evidence
  * unconfirmed/ambiguous edge   -> candidate RETAINED with boundary context
  * missing PTS / low coverage   -> explicit gaps, quality gate reason, never
                                    a silently-passing document
  * repeated runs                -> byte-identical core JSON
No LLM / no GPU / no network.
"""
from __future__ import annotations

import json
import unittest

import numpy as np

from omintel.event_measurements import compute_luminance_sequence
from omintel.event_models import BoundarySupport, ShotSpan, validate_events_document
from omintel.event_extractors import (
    DEFAULT_PARAMS,
    extract_brightness_events,
    find_brightness_candidates,
    merge_runs,
)

PTS_STEP = 0.037  # deliberately not 1/fps


def _frames(n):
    return [{"frame_no": i, "pts_time_s": f"{i * PTS_STEP:.6f}"} for i in range(n)]


def _gray(v, size=8):
    return np.full((size, size), v, dtype=np.uint8)


def _meas(values):
    frames = _frames(len(values))
    return compute_luminance_sequence(frames, [_gray(v) for v in values])


def _shot(sid="S0001", a=0, b=None, n=40, start_bd=None, end_bd=None):
    b = b if b is not None else n
    return ShotSpan(shot_id=sid, start_frame_no=a, end_frame_no=b,
                    start_pts_s=a * PTS_STEP, end_pts_s=(b - 1) * PTS_STEP + 0.001,
                    start_boundary=start_bd, end_boundary=end_bd)


def _doc_json(d):
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


class ScreeningTests(unittest.TestCase):
    def test_single_step_locates_correct_pair_and_pts(self):
        vals = [100] * 10 + [180] * 10
        m = _meas(vals)
        cand, an = find_brightness_candidates(m)
        self.assertEqual(len(cand), 1)
        c = cand[0]
        self.assertEqual((c["pre_frame_no"], c["post_frame_no"]), (9, 10))
        self.assertAlmostEqual(c["delta_mean_luma"], 80.0, places=6)
        self.assertAlmostEqual(c["post_pts_s"], 10 * PTS_STEP, places=6)
        self.assertFalse(c["boundary_ambiguous"])

    def test_encoding_noise_below_threshold_produces_nothing(self):
        vals = [120, 120, 121, 120, 121, 121, 120]  # |delta| <= 1
        cand, _ = find_brightness_candidates(_meas(vals))
        self.assertEqual(cand, [])

    def test_ramp_does_not_become_a_chain_of_jumps(self):
        vals = [50 + i for i in range(30)]  # delta=1/frame, cumulative 29
        cand, _ = find_brightness_candidates(_meas(vals))
        self.assertEqual(cand, [])

    def test_threshold_is_param_not_hardcoded_truth(self):
        vals = [100] * 5 + [106] * 5  # delta 6: below default 8, above 5
        self.assertEqual(find_brightness_candidates(_meas(vals))[0], [])
        cand, an = find_brightness_candidates(_meas(vals), params={"delta_threshold": 5.0})
        self.assertEqual(len(cand), 1)
        self.assertEqual(an["params"]["delta_threshold"], 5.0)

    def test_consecutive_large_deltas_merge_deterministically(self):
        # two adjacent big steps (flash-like) must become ONE run
        vals = [100, 100, 140, 180, 180, 180, 100, 100]
        m = _meas(vals)
        cand, _ = find_brightness_candidates(m)
        self.assertEqual(len(cand), 3)  # deltas at pairs 1,2 (+40,+40) and 6(-80)
        runs = merge_runs(cand, m, threshold=DEFAULT_PARAMS["delta_threshold"], merge_gap_frames=1)
        self.assertEqual([len(r) for r in runs], [2, 1])

    def test_sustained_excursion_merges_into_single_run(self):
        # monotone multi-frame brightening: every delta qualifies -> ONE run
        vals = [80] * 5 + [95, 110, 125, 140] + [140] * 5
        m = _meas(vals)
        cand, _ = find_brightness_candidates(m)
        runs = merge_runs(cand, m, threshold=DEFAULT_PARAMS["delta_threshold"], merge_gap_frames=1)
        self.assertEqual([len(r) for r in runs], [len(cand)])


class AttributionTests(unittest.TestCase):
    def test_confirmed_hard_cut_neighbor_is_suppressed_not_claimed(self):
        bd = BoundarySupport(boundary_id="B0001", review_status="CONFIRMED",
                             support_level="manual_review", support_count=2,
                             pts_s=5 * PTS_STEP, frame_no=5)
        shot = _shot("S0002", a=5, n=20, start_bd=bd)
        shots = [_shot("S0001", a=0, b=5, n=20), shot]
        vals = [180] * 5 + [60] * 15  # cut at frame 5: big drop INTO next shot
        doc = extract_brightness_events(_meas(vals), shots,
                                        video_id="v", source_sha256="ab" * 32)
        self.assertEqual(doc["events"], [])
        self.assertEqual(len(doc["boundary_attributed_transitions"]), 1)
        sup = doc["boundary_attributed_transitions"][0]
        self.assertEqual(sup["reason"], "adjacent_to_confirmed_hard_cut")
        self.assertEqual(sup["boundary_support"]["review_status"], "CONFIRMED")
        validate_events_document(doc)

    def test_unconfirmed_boundary_keeps_candidate_with_context(self):
        bd = BoundarySupport(boundary_id="B0001", review_status="UNCONFIRMED",
                             support_level="single_detector", support_count=1,
                             pts_s=5 * PTS_STEP, frame_no=5)
        shots = [_shot("S0001", a=0, b=5, n=20),
                 _shot("S0002", a=5, n=20, start_bd=bd)]
        vals = [180] * 5 + [60] * 15
        ident = {i: ("e" * 63 + str(i % 10)) for i in range(20)}
        doc = extract_brightness_events(_meas(vals), shots,
                                        video_id="v", source_sha256="ab" * 32,
                                        sha_by_fn=ident)
        self.assertEqual(len(doc["events"]), 1)
        ev = doc["events"][0]
        self.assertEqual(ev["status"], "CANDIDATE")
        self.assertIsNone(ev["confidence"])
        ctx = ev["evidence"]["boundary_context"]
        self.assertEqual(ctx["start_boundary"]["review_status"], "UNCONFIRMED")
        # half-open ownership: post-cut frame 5 belongs to S0002 only
        self.assertEqual(ev["shot_id"], "S0002")
        self.assertGreaterEqual(ev["interval_frames"]["start"], 5)
        validate_events_document(doc)

    def test_run_spanning_shot_edge_is_split_never_crosses(self):
        shots = [_shot("S0001", a=0, b=10, n=30), _shot("S0002", a=10, n=30)]
        # sustained multi-frame brightening crossing frame 10
        vals = [80] * 8 + [100, 130, 160, 190] + [190] * 6
        ident = {i: ("d" * 63 + str(i % 10)) for i in range(len(vals))}
        doc = extract_brightness_events(_meas(vals), shots,
                                        video_id="v", source_sha256="ab" * 32,
                                        sha_by_fn=ident)
        for ev in doc["events"]:
            sh = next(s for s in doc["shots"] if s["shot_id"] == ev["shot_id"])
            self.assertLessEqual(sh["start_frame_no"], ev["interval_frames"]["start"])
            self.assertLessEqual(ev["interval_frames"]["end"], sh["end_frame_no"])
        validate_events_document(doc)

    def test_transition_outside_any_shot_becomes_explicit_gap(self):
        shots = [_shot("S0001", a=0, b=8, n=20)]  # frames >=8 unattributed
        vals = [180] * 10 + [60] * 10  # jump at pair 9->10, outside all shots
        doc = extract_brightness_events(_meas(vals), shots,
                                        video_id="v", source_sha256="ab" * 32)
        self.assertEqual(doc["events"], [])
        self.assertEqual(len(doc["unattributed_transitions"]), 1)
        g = doc["unattributed_transitions"][0]
        self.assertEqual(g["reason"], "boundary_ambiguity")
        validate_events_document(doc)


class UncertaintyAndQualityTests(unittest.TestCase):
    def test_missing_pts_on_delta_carries_covering_gap(self):
        frames = _frames(20)
        frames[10]["pts_time_s"] = ""
        images = [_gray(100)] * 10 + [_gray(180)] * 10
        m = compute_luminance_sequence(frames, images)
        shots = [_shot("S0001", a=0, b=20)]
        ident = {i: ("a" * 63 + str(i % 10)) for i in range(20)}
        doc = extract_brightness_events(m, shots, video_id="v", source_sha256="ab" * 32,
                                        sha_by_fn=ident)
        ev = doc["events"][0]
        gap_fns = {(g["frame_range"]["start"], g["frame_range"]["end"]) for g in ev["evidence"]["gaps"]}
        self.assertIn((10, 11), gap_fns)
        # every frame_ref without PTS must be covered by a gap (validator enforces)
        validate_events_document(doc)

    def test_low_coverage_forces_quality_reason(self):
        frames = _frames(10)
        images = [_gray(100)] * 5 + [None] * 4 + [_gray(100)]
        m = compute_luminance_sequence(frames, images)
        self.assertLess(m["coverage_ratio"], DEFAULT_PARAMS["min_coverage_ratio"])
        doc = extract_brightness_events(m, [_shot("S0001", a=0, b=10)],
                                        video_id="v", source_sha256="ab" * 32)
        self.assertEqual(doc["quality_gate"]["status"], "REVIEW")
        self.assertIn("coverage_ratio", doc["quality_gate"]["reason"])
        validate_events_document(doc)

    def test_empty_result_still_review_not_pass(self):
        doc = extract_brightness_events(_meas([120] * 12), [_shot("S0001", a=0, b=12)],
                                        video_id="v", source_sha256="ab" * 32)
        self.assertEqual(doc["events"], [])
        self.assertEqual(doc["quality_gate"]["status"], "REVIEW")
        self.assertTrue(doc["quality_gate"]["reason"])
        validate_events_document(doc)


class DeterminismTests(unittest.TestCase):
    def test_repeated_extraction_is_byte_identical(self):
        vals = [100] * 6 + [170] * 6 + [170, 100, 100, 100, 100, 100]
        shots = [_shot("S0001", a=0, b=len(vals))]
        ident = {i: ("f" * 63 + str(i % 10)) for i in range(len(vals))}
        mk = lambda: _doc_json(extract_brightness_events(
            _meas(vals), shots, video_id="v", source_sha256="ab" * 32, sha_by_fn=ident))
        a, b = mk(), mk()
        self.assertEqual(a, b)
        # and params/version are inside the hashed content (provenance)
        d = json.loads(a)
        self.assertEqual(d["algorithm_version"], "1.0.0")

    def test_measurement_digest_links_doc_to_input(self):
        m = _meas([100] * 6 + [170] * 6)
        ident = {i: ("f" * 63 + str(i % 10)) for i in range(12)}
        doc = extract_brightness_events(m, [_shot("S0001", a=0, b=12)],
                                        video_id="v", source_sha256="ab" * 32,
                                        sha_by_fn=ident)
        self.assertEqual(doc["measurement_provenance"]["input_digest"], m["input_digest"])


if __name__ == "__main__":
    unittest.main()
