"""L1.5 Commit E (E2b): peak_motion candidate-criteria contract tests.

Written BEFORE the implementation, per the approved order: freeze criteria
tests first, then the pure extractor.  Every test pins one clause of the
v1.0.0 contract revised in review:

  R1  prominence window EXCLUDES the plateau itself and only uses valid,
      unsuppressed pairs; a flat top (consecutive equal values that all
      satisfy the peak condition) yields exactly ONE representative peak at
      the LEFT-MIDDLE index -- deterministic regardless of traversal order.
  R2  frame-pair -> event interval uses POST-frame semantics: active pair
      range [a..b] covers post frames a+1..b+1, so the half-open event
      interval is [frame_no[a+1], frame_no[b+1]+1); it must contain both
      the first and the last post frame, stay strictly right-open, never
      cross a ShotSpan edge and never span a non-contiguous frame-number
      gap or missing-data hole.
  R3  confirmed-cut semantics come ONLY from BoundarySupport.is_hard_cut()
      (review_status == CONFIRMED).  No new enums.  Ambiguity is recorded
      as review_reasons / boundary_context, NOT disguised as a CoverageGap
      when measurements are complete.
  R4  a candidate whose strongest peak has no authoritative PTS is NOT
      emitted as an event; it becomes an explicit audit entry + gap
      (E1b principle: no fabricated timestamps).
  Plus the mandated acceptance matrix: static, constant translation,
  sudden increase/decrease, multi-peak burst merged into one candidate,
  two independent bursts kept apart, hard cut straddling pair suppressed
  as peak while both sides may form their own candidates, brightness
  step with static content retained-but-flagged, bad frames / missing PTS,
  parameter sensitivity recorded in provenance, byte-identical reruns.

No LLM, no GPU, no network, no disk writes.
"""
from __future__ import annotations

import json
import unittest

import numpy as np

from omintel.event_models import (
    BoundarySupport,
    ShotSpan,
    validate_events_document,
)
from omintel.event_motion import compute_motion_sequence
from omintel.event_extractors_motion import extract_peak_motion_events

PTS_STEP = 0.037  # deliberately NOT representable as 1/fps: pins PTS authority


# --------------------------------------------------------------------------
# synthetic-material helpers (drive the REAL E2a measurement path)
# --------------------------------------------------------------------------

def _frames(n: int):
    return [{"frame_no": i, "pts_time_s": f"{i * PTS_STEP:.6f}"} for i in range(n)]


def _scene_frames(n: int, size=(16, 16)):
    """Static scene: uniform gray."""
    return [np.full(size, 100, dtype=np.uint8) for _ in range(n)]


def _motion_frames(n: int, moving: set[int], size=(32, 32)):
    """Frames in `moving` show a block shifted by 4 px vs the resting layout.

    A contiguous run of moving frames produces the canonical symmetric
    burst shape measured in E2a: entry spike, ~zero plateau (block static
    while "moving"), exit spike.
    """
    imgs = []
    for i in range(n):
        g = np.zeros(size, dtype=np.uint8)
        if i in moving:
            g[4:20, 8:16] = 220      # moved layout (static within the run)
        else:
            g[4:20, 4:12] = 220      # resting layout
        imgs.append(g)
    return imgs


def _translation_frames(n: int, shift_per_frame: int = 2, size=(32, 32)):
    """A bright block slides right every frame: sustained uniform motion."""
    imgs = []
    for i in range(n):
        g = np.zeros(size, dtype=np.uint8)
        x = (i * shift_per_frame) % (size[1] - 8)
        g[4:20, x:x + 8] = 220
        imgs.append(g)
    return imgs


def _two_burst_frames(n: int, b1: set[int], b2: set[int], size=(32, 32)):
    """Two independent motion bursts separated by resting frames."""
    imgs = []
    for i in range(n):
        g = np.zeros(size, dtype=np.uint8)
        if i in b1:
            g[4:20, 8:16] = 220       # burst 1 layout
        elif i in b2:
            g[4:20, 8:16] = 90        # burst 2 layout (different amplitude)
        else:
            g[4:20, 4:12] = 220       # resting layout
        imgs.append(g)
    return imgs


def _hard_cut_frames(n: int, cut_at: int, size=(32, 32)):
    """Identical structure, opposite polarity across frame `cut_at`.

    The crossing pair (cut_at-1, cut_at) shows a huge diff; each side is
    internally static.  Additionally BOTH flanking pairs (cut_at-2 and
    cut_at+1) carry the same large magnitude, so under any window that
    includes them the crossing pair cannot win prominence on its own --
    suppression of the cross-cut pair is therefore provably STRUCTURAL
    (confirmed hard cut via is_hard_cut()), not numeric thresholding.
    """
    imgs = []
    for i in range(n):
        g = np.zeros(size, dtype=np.uint8)
        if i == cut_at - 2 or i >= cut_at + 1:
            # shifted one extra block -> big diff vs the plain side layout
            if i < cut_at:
                g[4:20, 4:12] = 220
                g[24:28, 24:28] = 220
            else:
                g[4:20, 8:16] = 220
                g[24:28, 24:28] = 220
        elif i < cut_at:
            g[4:20, 4:12] = 220
        else:
            g[4:20, 8:16] = 220
        imgs.append(g)
    return imgs


def _brightness_step_static_content(n: int, step_at: int, size=(16, 16)):
    """Same pixels before/after step_at, just brighter: content is static."""
    imgs = []
    for i in range(n):
        v = 100 if i < step_at else 200
        g = np.full(size, v, dtype=np.uint8)
        g[4:8, 4:8] = v - 60          # identical dark square every frame
        imgs.append(g)
    return imgs


def _shot(sid="S0001", a=0, b=None, n=40, start_bd=None, end_bd=None):
    b = b if b is not None else n
    return ShotSpan(shot_id=sid, start_frame_no=a, end_frame_no=b,
                    start_pts_s=a * PTS_STEP, end_pts_s=(b - 1) * PTS_STEP + 0.001,
                    start_boundary=start_bd, end_boundary=end_bd)


def _bd(bid, status, level, fn):
    return BoundarySupport(boundary_id=bid, review_status=status,
                           support_level=level, support_count=1,
                           pts_s=fn * PTS_STEP, frame_no=fn, note=None)


def _meas(frames_rows, images, boundary_frame_nos=None):
    return compute_motion_sequence(frames_rows, images,
                                   boundary_frame_nos=boundary_frame_nos)


def _doc_json(d):
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


BASE_PARAMS = {"abs_threshold": 0.5, "prominence_ratio": 0.5,
               "window_frames": 3, "merge_gap_frames": 2}


def _extract(measurement, shots, params=None, sha_by_fn=None,
             video_id="vid_test", source_sha256="0" * 64):
    p = dict(BASE_PARAMS)
    if params:
        p.update(params)
    return extract_peak_motion_events(
        measurement, shots,
        video_id=video_id, source_sha256=source_sha256, params=p,
        sha_by_fn=sha_by_fn)


def _frame_no_of_pair(m, pair_index):
    """Post frame number of a pair index in a measurement record."""
    return m["frame_pairs"][pair_index]["frame_no"]


# ==========================================================================
# 1. Static / stable platform: no peaks, no events
# ==========================================================================

class TestStaticAndPlatform(unittest.TestCase):
    def test_static_scene_yields_no_candidates(self):
        m = _meas(_frames(8), _scene_frames(8))
        doc = _extract(m, [_shot(n=8)])
        self.assertEqual(doc["events"], [])
        validate_events_document(doc)

    def test_constant_translation_platform_does_not_spam_peaks(self):
        # sustained uniform motion: every pair equally nonzero -> no pair is
        # locally prominent; at most ONE structural onset peak, never a chain
        imgs = _translation_frames(8, shift_per_frame=2)
        m = _meas(_frames(8), imgs)
        means = [pr["mean_abs_diff"] for pr in m["frame_pairs"]]
        self.assertTrue(all(x > 0 for x in means), means)
        doc = _extract(m, [_shot(n=8)])
        self.assertLessEqual(len(doc["events"]), 1,
                             f"stable platform must not spam peaks: {means}")
        if doc["events"]:
            ev = doc["events"][0]
            iv = ev["interval_frames"]
            self.assertLessEqual(iv["end"] - iv["start"], 2,
                                 "any residual peak must be point-sized (onset), "
                                 "not a shot-wide activity interval")


# ==========================================================================
# 2. Sudden increase / decrease: correct localization, PTS authority
# ==========================================================================

class TestLocalization(unittest.TestCase):
    def test_onset_burst_locates_entry_peak_and_post_frame_interval(self):
        # rest 0..2, burst 3..5, rest 6..9  -> spikes at pairs 2 and 5
        imgs = _motion_frames(10, {3, 4, 5})
        m = _meas(_frames(10), imgs)
        doc = _extract(m, [_shot(n=10)])
        self.assertEqual(len(doc["events"]), 1, doc["events"])
        ev = doc["events"][0]
        iv = ev["interval_frames"]
        # R2 post-frame semantics: active pairs {2,3,4,5} cover post frames
        # 3..6 -> half-open [3, 7): FIRST change's post frame 3 IN, LAST
        # post frame 6 IN, endpoint strictly right-open.
        self.assertEqual((iv["start"], iv["end"]), (3, 7))
        self.assertGreaterEqual(iv["start"], 3)
        self.assertLess(iv["end"] - 1, 7)
        # peak time comes from the authoritative record (pair 2 -> frame 3)
        self.assertAlmostEqual(ev["peak_pts_s"], 3 * PTS_STEP, places=6)
        meas = ev["evidence"]["measurements"]
        self.assertIn("peaks", meas)
        self.assertEqual([p["post_frame_no"] for p in meas["peaks"]], [3, 6])
        # interval_pts endpoints are real measured PTS values, not fps math
        self.assertAlmostEqual(ev["interval_pts"]["start"], 3 * PTS_STEP, places=6)
        self.assertAlmostEqual(ev["interval_pts"]["end"], 6 * PTS_STEP, places=6)

    def test_sustained_excursion_is_one_candidate_with_multiple_peaks(self):
        imgs = _motion_frames(12, {4, 5, 6, 7})
        m = _meas(_frames(12), imgs)
        doc = _extract(m, [_shot(n=12)], params={"merge_gap_frames": 3})
        self.assertEqual(len(doc["events"]), 1)
        peaks = doc["events"][0]["evidence"]["measurements"]["peaks"]
        self.assertGreaterEqual(len(peaks), 2,
                                "entry AND exit spikes must be retained as evidence")
        iv = doc["events"][0]["interval_frames"]
        # active pairs 3..7 (plateau zeros bridged at merge_gap_frames=3)
        # -> post frames 4..8 -> half-open [4, 9)
        self.assertEqual((iv["start"], iv["end"]), (4, 9))


# ==========================================================================
# 3. Flat-top plateau: exactly one representative peak, left-middle rule (R1)
# ==========================================================================

class TestPlateauUniqueness(unittest.TestCase):
    def test_plateau_of_equal_values_produces_single_left_middle_peak(self):
        # synthetic series pinning the rule; plateau at pair indices 2..5
        # (length 4, EVEN) -> deterministic representative = LEFT MIDDLE = 3
        vals = [0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0]
        m = _synth_motion_measure(vals, n_frames=9)
        doc = _extract(m, [_shot(n=9)],
                       params={"abs_threshold": 0.5, "prominence_ratio": 0.5})
        self.assertEqual(len(doc["events"]), 1)
        peaks = doc["events"][0]["evidence"]["measurements"]["peaks"]
        self.assertEqual(len(peaks), 1,
                         "a flat top satisfying the peak condition is ONE peak")
        self.assertEqual(peaks[0]["pair_index"], 3)
        # representative position maps through the REAL record: pair 3's post
        # frame is frame 4 (contiguous synthetic frames)
        self.assertEqual(peaks[0]["post_frame_no"], _frame_no_of_pair(m, 3))
        self.assertEqual(peaks[0]["post_frame_no"], 4)

    def test_odd_length_plateau_uses_middle_index(self):
        # plateau at pair indices 2..4 (length 3, ODD) -> middle = 3
        vals = [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0]
        m = _synth_motion_measure(vals, n_frames=8)
        doc = _extract(m, [_shot(n=8)],
                       params={"abs_threshold": 0.5, "prominence_ratio": 0.5})
        peaks = doc["events"][0]["evidence"]["measurements"]["peaks"]
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0]["pair_index"], 3)


# ==========================================================================
# 4. Two independent bursts: no over-wide merge
# ==========================================================================

class TestIndependentBursts(unittest.TestCase):
    def test_two_separated_bursts_stay_two_candidates(self):
        imgs = _two_burst_frames(14, {3, 4}, {9, 10})
        m = _meas(_frames(14), imgs)
        doc = _extract(m, [_shot(n=14)])
        self.assertEqual(len(doc["events"]), 2,
                         "rest plateau between bursts must prevent merging")
        ivs = sorted((e["interval_frames"]["start"], e["interval_frames"]["end"])
                     for e in doc["events"])
        self.assertEqual(ivs[0], (3, 6))   # burst1: pairs 2..5 -> post 3..6
        self.assertEqual(ivs[1], (9, 12))  # burst2: pairs 8..11 -> post 9..12

    def test_deep_valley_splits_runs_bridge_rule_is_deterministic(self):
        # plateau pair indices 1..5; index 3 is the valley.  A valley BELOW
        # the bridge floor (abs_threshold/2) splits the run into two events;
        # a valley AT/ABOVE it bridges -> one event.
        vals = [0.0, 2.0, 2.0, 0.4, 2.0, 2.0, 0.0]
        m = _synth_motion_measure(vals, n_frames=8)
        doc = _extract(m, [_shot(n=8)], params={"merge_gap_frames": 2})
        self.assertEqual(len(doc["events"]), 2)
        vals2 = [0.0, 2.0, 2.0, 1.0, 2.0, 2.0, 0.0]  # 1.0 >= abs_thr/2 -> bridged
        m2 = _synth_motion_measure(vals2, n_frames=8)
        doc2 = _extract(m2, [_shot(n=8)], params={"merge_gap_frames": 2})
        self.assertEqual(len(doc2["events"]), 1)


# ==========================================================================
# 5. Hard-cut handling (R3): structural suppression via is_hard_cut()
# ==========================================================================

class TestHardCutPolicy(unittest.TestCase):
    def test_confirmed_cut_crossing_pair_never_becomes_a_peak(self):
        cut = 5
        imgs = _hard_cut_frames(10, cut)
        m = _meas(_frames(10), imgs, boundary_frame_nos=[cut])
        # sanity: the crossing pair really is the global maximum
        means = [pr["mean_abs_diff"] for pr in m["frame_pairs"]]
        self.assertEqual(int(np.argmax(means)), cut - 1)
        shotA = _shot("S0001", 0, cut, n=10)
        shotB = _shot("S0002", cut, 10, n=10,
                      start_bd=_bd("B0001", "CONFIRMED", "manual_review", cut))
        doc = _extract(m, [shotA, shotB])
        for ev in doc["events"]:
            for pk in ev["evidence"]["measurements"]["peaks"]:
                self.assertNotEqual(pk["pair_index"], cut - 1,
                                    "cross-cut pair must be peak-ineligible")
        # each side has NO internal motion here -> zero events, but the
        # suppressed peak must survive in the audit list
        sup = doc["audit"]["suppressed_peaks"]
        self.assertEqual([s["pair_index"] for s in sup], [cut - 1])
        self.assertEqual(sup[0]["reason"], "adjacent_to_confirmed_hard_cut")
        self.assertEqual(sup[0]["boundary_support"]["review_status"], "CONFIRMED")

    def test_both_sides_of_cut_keep_their_own_evidence_if_motion_exists(self):
        # motion burst inside shot A (frames 2..3) and another inside shot B
        # (frames 8..9), cut at frame 6.  The crossing pair (5,6) reads
        # 6.875 -- above abs_threshold but NOT locally prominent (neighbors
        # are 0), so it never becomes a peak at all: the structural
        # suppression rule is pinned by the dedicated confirmed-cut test.
        # Here we require BOTH sides to keep their own candidates.
        cut = 6
        imgs = _two_burst_frames(12, {2, 3}, {8, 9})
        for i in range(cut, 12):
            imgs[i] = np.roll(imgs[i], 1, axis=1)
        m = _meas(_frames(12), imgs, boundary_frame_nos=[cut])
        shotA = _shot("S0001", 0, cut, n=12)
        shotB = _shot("S0002", cut, 12, n=12,
                      start_bd=_bd("B0001", "CONFIRMED", "manual_review", cut))
        doc = _extract(m, [shotA, shotB])
        ids = {e["shot_id"] for e in doc["events"]}
        self.assertIn("S0001", ids)
        self.assertIn("S0002", ids)
        for e in doc["events"]:
            iv = e["interval_frames"]
            if e["shot_id"] == "S0001":
                self.assertLessEqual(iv["end"], cut)
            else:
                self.assertGreaterEqual(iv["start"], cut)

    def test_activity_split_at_confirmed_cut_never_span_it(self):
        # continuous activity pairs 2..6 crossing cut at frame 5.  The
        # crossing pair (idx 4 = (4,5)) is peak-INELIGIBLE under a confirmed
        # cut, so the two sides each keep their own evidence and become two
        # shot-local candidates -- never one spanning event.
        vals = [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0]
        m = _synth_motion_measure(vals, n_frames=9, boundary_frame_nos=[5])
        shotA = _shot("S0001", 0, 5, n=9)
        shotB = _shot("S0002", 5, 9, n=9,
                      start_bd=_bd("B0001", "CONFIRMED", "manual_review", 5))
        doc = _extract(m, [shotA, shotB])
        self.assertEqual(len(doc["events"]), 2, doc["events"])
        by_shot = {e["shot_id"]: e for e in doc["events"]}
        self.assertEqual(by_shot["S0001"]["interval_frames"],
                         {"domain": "frame_no", "start": 2, "end": 5})
        self.assertEqual(by_shot["S0002"]["interval_frames"],
                         {"domain": "frame_no", "start": 5, "end": 8})
        # neither event crosses the cut; right-open endpoints touch it exactly
        self.assertLessEqual(by_shot["S0001"]["interval_frames"]["end"], 5)
        self.assertGreaterEqual(by_shot["S0002"]["interval_frames"]["start"], 5)
        validate_events_document(doc)

    def test_unconfirmed_boundary_does_not_suppress_but_flags_review(self):
        cut = 5
        imgs = _hard_cut_frames(10, cut)
        m = _meas(_frames(10), imgs, boundary_frame_nos=[cut])
        shotA = _shot("S0001", 0, cut, n=10)
        shotB = _shot("S0002", cut, 10, n=10,
                      start_bd=_bd("B0001", "REVIEW_REQUIRED", "agreement", cut))
        doc = _extract(m, [shotA, shotB])
        # NOT suppressed: no confirmed verdict exists (is_hard_cut False)
        sup = doc["audit"].get("suppressed_peaks", [])
        self.assertEqual([s["pair_index"] for s in sup], [])
        # but the big change is attributed to the NEXT shot (post semantics)
        # and flagged as needing review -- never presented as confirmed cut
        self.assertEqual(len(doc["events"]), 1)
        ev = doc["events"][0]
        self.assertEqual(ev["shot_id"], "S0002")
        self.assertIn("unconfirmed_boundary_neighbor",
                      ev["evidence"]["measurements"]["review_reasons"])
        bc = ev["evidence"].get("boundary_context") or {}
        self.assertIn("start_boundary", bc)
        self.assertEqual(bc["start_boundary"]["review_status"], "REVIEW_REQUIRED")
        # ambiguity is review context, NOT disguised as a coverage gap
        # (measurements are complete here)
        for g in ev["evidence"]["gaps"]:
            self.assertNotEqual(g["reason"], "boundary_ambiguity")


# ==========================================================================
# 6. Brightness step with static content: retained + flagged (approved item 2)
# ==========================================================================

class TestBrightnessStepAmbiguity(unittest.TestCase):
    def test_static_content_step_is_flagged_not_reclassified(self):
        step = 5
        imgs = _brightness_step_static_content(10, step)
        m = _meas(_frames(10), imgs)
        doc = _extract(m, [_shot(n=10)])
        self.assertEqual(len(doc["events"]), 1,
                         "motion detector keeps the candidate; classification "
                         "belongs downstream")
        meas = doc["events"][0]["evidence"]["measurements"]
        self.assertIn("possible_brightness_transition", meas)
        self.assertTrue(meas["possible_brightness_transition"])
        self.assertIn("possible_brightness_transition",
                      meas["review_reasons"])


# ==========================================================================
# 7. Missing data (R4): peak without PTS -> audit + gap, never a fake event
# ==========================================================================

class TestMissingData(unittest.TestCase):
    def test_peak_without_pts_is_not_emitted_as_event(self):
        rows = _frames(10)
        rows[3]["pts_time_s"] = ""   # post frame of the entry-spike pair lost PTS
        imgs = _motion_frames(10, {3, 4, 5})
        m = _meas(rows, imgs)
        doc = _extract(m, [_shot(n=10)])
        self.assertEqual(doc["events"], [],
                         "strongest peak lacking authoritative PTS cannot be an event")
        aud = doc["audit"]["candidates_without_peak_pts"]
        self.assertEqual(len(aud), 1)
        self.assertEqual(aud[0]["peak_post_frame_no"], 3)
        reasons = {g["reason"] for g in aud[0]["gaps"]}
        self.assertIn("missing_pts", reasons)
        validate_events_document(doc)

    def test_invalid_interior_pair_splits_or_truncates_run(self):
        # burst frames 3..7 (pairs 2..7 active); frame 5 undecodable ->
        # pairs 4 and 5 invalid.  Nothing may silently span the hole: the
        # run must split into two shot-local candidates on frames {3,4}
        # and {6,7}.
        imgs = _motion_frames(10, {3, 4, 5, 6, 7})
        imgs[5] = None               # decode failure mid-burst
        m = _meas(_frames(10), imgs)
        doc = _extract(m, [_shot(n=10)])
        self.assertEqual(len(doc["events"]), 2, doc["events"])
        ivs = sorted((e["interval_frames"]["start"], e["interval_frames"]["end"])
                     for e in doc["events"])
        self.assertEqual(ivs, [(3, 5), (6, 8)])
        # frame 5 (the unmeasurable one) is covered by NEITHER interval
        for ev in doc["events"]:
            iv = ev["interval_frames"]
            covered = set(range(iv["start"], iv["end"]))
            self.assertNotIn(5, covered,
                             "event interval cannot silently cross an unmeasurable frame")
        gaps = doc["audit"].get("measurement_gaps", [])
        self.assertTrue(any(g["reason"] == "measurement_unavailable" for g in gaps))

    def test_low_coverage_forces_quality_reason_not_pass(self):
        imgs = _motion_frames(10, {3, 4, 5})
        for i in (6, 7, 8):
            imgs[i] = None
        m = _meas(_frames(10), imgs)
        doc = _extract(m, [_shot(n=10)])
        self.assertEqual(doc["quality_gate"]["status"], "REVIEW")
        self.assertIsNotNone(doc["quality_gate"].get("reason"))


# ==========================================================================
# 8. Parameters & provenance: thresholds are declared policy, not truth
# ==========================================================================

class TestParametersProvenance(unittest.TestCase):
    def test_result_changes_only_via_explicit_params(self):
        imgs = _motion_frames(10, {3, 4, 5})
        m = _meas(_frames(10), imgs)
        loose = _extract(m, [_shot(n=10)], params={"abs_threshold": 0.05})
        strict = _extract(m, [_shot(n=10)], params={"abs_threshold": 999.0})
        self.assertEqual(len(strict["events"]), 0)
        self.assertGreaterEqual(len(loose["events"]), 1)
        prov = loose["measurement_provenance"]
        self.assertEqual(prov["algorithm_params"]["abs_threshold"], 0.05)
        self.assertEqual(prov["algorithm_params"]["prominence_ratio"], 0.5)

    def test_provenance_links_doc_to_measurement_digest(self):
        imgs = _motion_frames(10, {3, 4, 5})
        m = _meas(_frames(10), imgs)
        doc = _extract(m, [_shot(n=10)])
        self.assertEqual(doc["measurement_provenance"]["input_digest"],
                         m["input_digest"])


# ==========================================================================
# 9. Determinism + lifecycle invariants
# ==========================================================================

class TestDeterminismLifecycle(unittest.TestCase):
    def test_repeated_extraction_is_byte_identical(self):
        imgs = _two_burst_frames(14, {3, 4}, {9, 10})
        m = _meas(_frames(14), imgs)
        d1 = _doc_json(_extract(m, [_shot(n=14)]))
        d2 = _doc_json(_extract(m, [_shot(n=14)]))
        self.assertEqual(d1, d2)

    def test_all_events_are_candidates_with_null_confidence(self):
        imgs = _motion_frames(12, {4, 5, 6, 7})
        m = _meas(_frames(12), imgs)
        doc = _extract(m, [_shot(n=12)])
        self.assertTrue(doc["events"])
        for ev in doc["events"]:
            self.assertEqual(ev["status"], "CANDIDATE")
            self.assertIsNone(ev["confidence"])
            self.assertTrue(ev["confidence_note"])
        self.assertEqual(doc["quality_gate"]["status"], "REVIEW")

    def test_empty_result_is_valid_review_not_pass(self):
        m = _meas(_frames(8), _scene_frames(8))
        doc = _extract(m, [_shot(n=8)])
        self.assertEqual(doc["events"], [])
        self.assertEqual(doc["quality_gate"]["status"], "REVIEW")
        validate_events_document(doc)


# --------------------------------------------------------------------------
# synthetic-series helper: build a VALID measurement record with exact
# mean_abs_diff values (for pinning criteria rules independently of pixel
# geometry).  max/ratio filled deterministically; all pairs valid.
# --------------------------------------------------------------------------

def _synth_motion_measure(pair_means: list[float], n_frames: int,
                          boundary_frame_nos=None):
    assert len(pair_means) == n_frames - 1
    rows = _frames(n_frames)
    # flat gray images: real metrics would be 0, so we override the record
    # fields while keeping the E2a schema byte-for-byte.
    base = compute_motion_sequence(rows, _scene_frames(n_frames),
                                   boundary_frame_nos=boundary_frame_nos)
    bset = set(boundary_frame_nos or [])
    for pr, v in zip(base["frame_pairs"], pair_means):
        pr["mean_abs_diff"] = float(v)
        pr["max_abs_diff"] = float(v)
        pr["changed_pixel_ratio"] = float(v)
        pr["adjacent_to_boundary"] = pr["frame_no"] in bset
    return base


if __name__ == "__main__":
    unittest.main()
