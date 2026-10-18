"""E2a contract tests: frame-pair motion measurement sequence.

Scenarios per review checklist: static scene, constant translation, sudden
motion increase, sudden decrease, isolated peak, hard cut, brightness step
with static content (pins the mean_abs_diff limitation), bad frames, missing
PTS, coverage denominator, boundary adjacency definition, timestamp
semantics, determinism and provenance.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from omintel.event_motion import (
    ALGORITHM_VERSION,
    DEFAULT_PARAMS,
    METRIC_NAME,
    compute_motion_sequence,
)


def _frames(n: int, step_s: float = 0.037, gap_at: int | None = None):
    """L0-style rows on an authoritative PTS timeline (deliberately NOT
    frame_no/fps: step 0.037 s is not representable at 30 fps)."""
    rows = []
    for i in range(n):
        pts = "" if i == gap_at else f"{i * step_s:.6f}"
        rows.append({"frame_no": i, "pts_time_s": pts})
    return rows


def _static(n: int, size=(16, 16)):
    base = np.full(size, 100, dtype=np.uint8)
    return [base.copy() for _ in range(n)]


def _translated(n: int, shift_per_frame: int, size=(32, 32)):
    """A bright block slides right by `shift_per_frame` px each frame."""
    imgs = []
    for i in range(n):
        g = np.zeros(size, dtype=np.uint8)
        x = (i * shift_per_frame) % (size[1] - 8)
        g[8:24, x:x + 8] = 220
        imgs.append(g)
    return imgs


def _pair(rec, idx):
    return rec["frame_pairs"][idx]


# ---------------------------------------------------------------- scenarios

def test_static_scene_reads_zero_not_gaps():
    rec = compute_motion_sequence(_frames(5), _static(5))
    assert rec["coverage_ratio"] == 1.0
    assert rec["valid_pair_count"] == 4
    for pr in rec["frame_pairs"]:
        assert pr["valid"] is True
        assert pr["mean_abs_diff"] == 0.0
        assert pr["max_abs_diff"] == 0.0
        assert pr["changed_pixel_ratio"] == 0.0
    # static must never be recorded as a gap
    assert rec["gaps"] == []


def test_constant_translation_produces_uniform_positive_signal():
    rec = compute_motion_sequence(_frames(6), _translated(6, 2))
    means = [pr["mean_abs_diff"] for pr in rec["frame_pairs"]]
    assert all(m > 0 for m in means)
    spread = max(means) - min(means)
    assert spread < 0.05 * max(means), f"translation unstable: {means}"


def test_sudden_increase_and_decrease_are_visible_in_series():
    n = 9
    # static until frame 3; moving object in frames 4..6; static again 7..8.
    # Motion ACTIVE intervals: pairs 3 (transition into motion), 4, 5.
    # Pair 6 is the transition OUT of motion (still nonzero); pairs 7, 8 quiet.
    base = np.full((16, 16), 100, dtype=np.uint8)
    imgs = [base.copy() for _ in range(n)]
    for k, fn in enumerate(range(4, 7)):
        g = base.copy()
        g[:, (k * 3) : (k * 3) + 4] = 250
        imgs[fn] = g
    rec = compute_motion_sequence(_frames(n), imgs)
    means = [pr["mean_abs_diff"] for pr in rec["frame_pairs"]]
    # moving window = pairs 3..6 (transition-in, two sustained, transition-out)
    active = means[3:7]
    quiet = [means[0], means[1], means[2], means[7]]
    assert min(active) > max(quiet), f"increase/decrease not separable: {means}"
    assert all(q == 0.0 for q in quiet)


def test_isolated_peak_is_a_two_pair_burst_flanked_by_zeros():
    """A single deviant frame creates two adjacent spikes (in/out edges).

    E2a only measures; whether to merge the two edges into one candidate is
    an E2b judgment rule -- so here we pin the raw shape: burst >> flanks.
    """
    n = 7
    imgs = _static(n)
    h = imgs[3].copy()
    h[:, :] = 200  # one-frame flash of whole frame -> isolated diff burst
    imgs[3] = h
    rec = compute_motion_sequence(_frames(n), imgs)
    means = [pr["mean_abs_diff"] for pr in rec["frame_pairs"]]
    peak_idx = int(np.argmax(means))
    assert peak_idx in (2, 3)
    other_spike = 3 if peak_idx == 2 else 2
    assert means[other_spike] == means[peak_idx]      # symmetric in/out edges
    flanks = [v for i, v in enumerate(means) if i not in (2, 3)]
    assert max(flanks) == 0.0                          # everything else static
    assert means[peak_idx] == pytest.approx(100.0)     # |200-100| everywhere


def test_hard_cut_pair_is_flagged_and_values_kept():
    n = 6
    imgs = _static(n)
    cut = 3  # boundary frame number b=3: first frame of next shot
    imgs[cut:] = [np.full((16, 16), 240, dtype=np.uint8) for _ in range(n - cut)]
    rec = compute_motion_sequence(_frames(n), imgs, boundary_frame_nos=[cut])
    flags = [pr["adjacent_to_boundary"] for pr in rec["frame_pairs"]]
    # frozen definition: exactly the pair (b-1, b) is flagged
    assert flags == [False, False, True, False, False]
    pair = _pair(rec, cut - 1)
    assert pair["prev_frame_no"] == cut - 1 and pair["frame_no"] == cut
    # flagging must NOT zero or drop the measurement
    assert pair["valid"] is True and pair["mean_abs_diff"] > 100


def test_brightness_step_with_static_content_pins_metric_limitation():
    """Content does not move; only global brightness steps up.

    mean_abs_diff spikes anyway -- this test freezes the documented
    limitation that E2a measures picture change, not motion.  The three
    metrics stay unfused so E2b can combine them with boundary evidence.
    """
    n = 5
    imgs = [np.full((16, 16), 90, dtype=np.uint8) for _ in range(n)]
    imgs[3:] = [np.full((16, 16), 180, dtype=np.uint8) for _ in range(2)]
    rec = compute_motion_sequence(_frames(n), imgs)
    step_pair = _pair(rec, 2)
    assert step_pair["mean_abs_diff"] > 50      # spike despite zero motion
    assert step_pair["changed_pixel_ratio"] == 1.0
    # no fused confidence anywhere in the record
    flat = json.dumps(rec)
    for forbidden in ("confidence", "motion_score", "probability"):
        assert forbidden not in flat


def test_missing_pts_yields_gap_not_interpolated_timestamp():
    n = 5
    imgs = _static(n)
    rec = compute_motion_sequence(_frames(n, gap_at=3), imgs)
    # pairs 2 (cur=3) and 3 (prev=3) are affected; cur frame 3 lacks PTS
    p2 = _pair(rec, 2)
    assert p2["valid"] is False and p2["invalid_reason"] == "missing_pts"
    assert p2["pts_s"] is None                    # never fabricated
    assert p2["mean_abs_diff"] is None            # never zero-filled
    p3 = _pair(rec, 3)
    assert p3["valid"] is True                    # prev missing PTS but cur has it
    assert p3["pts_s"] == pytest.approx(4 * 0.037, abs=1e-9)
    assert any(g["reason"] == "missing_pts" for g in rec["gaps"])


def test_bad_frames_record_measurement_unavailable_and_lower_coverage():
    n = 6
    imgs = _static(n)
    imgs[2] = None                                # decode failure
    rec = compute_motion_sequence(_frames(n), imgs)
    # theoretical pairs = 5; invalid pairs touching frame 2 = pair1, pair2
    assert rec["pair_count"] == 5
    assert rec["valid_pair_count"] == 3
    assert rec["coverage_ratio"] == pytest.approx(3 / 5, abs=1e-9)
    reasons = {_pair(rec, i)["invalid_reason"] for i in (1, 2)}
    assert reasons == {"measurement_unavailable"}
    assert any(g["reason"] == "measurement_unavailable" for g in rec["gaps"])


def test_coverage_denominator_is_theoretical_pairs():
    n = 4
    imgs = _static(n)
    imgs[1] = None
    imgs[2] = None
    rec = compute_motion_sequence(_frames(n), imgs)
    # every pair touches a bad frame except none -> valid 0/3
    assert rec["pair_count"] == 3
    assert rec["valid_pair_count"] == 0
    assert rec["coverage_ratio"] == 0.0


def test_timestamp_semantics_use_current_frame_pts():
    rec = compute_motion_sequence(_frames(4), _static(4))
    for pr in rec["frame_pairs"]:
        expected = round(pr["frame_no"] * 0.037, 6)
        assert pr["pts_s"] == pytest.approx(expected, abs=1e-9)
    # explicitly NOT frame_no/fps: 0.037 != k/30 grid
    assert _pair(rec, 1)["pts_s"] != pytest.approx(1 / 30)


def test_length_mismatch_and_nonincreasing_frame_no_raise():
    with pytest.raises(ValueError):
        compute_motion_sequence(_frames(3), _static(4))
    with pytest.raises(ValueError):
        compute_motion_sequence(_frames(1), _static(1))
    bad = _frames(3)
    bad[2]["frame_no"] = 1                        # duplicate/decreasing order
    with pytest.raises(ValueError):
        compute_motion_sequence(bad, _static(3))
    imgs = _static(3)
    imgs[2] = np.zeros((8, 8), dtype=np.uint8)    # shape mismatch
    with pytest.raises(ValueError):
        compute_motion_sequence(_frames(3), imgs)


def test_determinism_byte_identical_serialization():
    n = 8
    imgs = _translated(n, 2)
    kw = dict(boundary_frame_nos=[5])
    r1 = compute_motion_sequence(_frames(n), imgs, **kw)
    r2 = compute_motion_sequence(_frames(n), imgs, **kw)
    j1 = json.dumps(r1, sort_keys=True)
    j2 = json.dumps(r2, sort_keys=True)
    assert j1 == j2
    assert r1["input_digest"] == r2["input_digest"]
    # different params -> different digest AND params recorded in provenance
    r3 = compute_motion_sequence(_frames(n), imgs, params={"pixel_change_threshold": 30}, **kw)
    assert r3["input_digest"] != r1["input_digest"]
    assert r3["provenance"]["params"]["pixel_change_threshold"] == 30
    assert r1["provenance"]["params"] == DEFAULT_PARAMS


def test_provenance_records_metric_identity_and_caveat():
    rec = compute_motion_sequence(_frames(3), _static(3))
    prov = rec["provenance"]
    assert prov["name"] == METRIC_NAME
    assert prov["algorithm_version"] == ALGORITHM_VERSION
    assert prov["params"] == DEFAULT_PARAMS
    assert "not a validated" in prov["notes"]     # pixel threshold caveat pinned
    assert rec["metric_name"] == METRIC_NAME


def test_gap_intervals_are_half_open_and_typed():
    n = 5
    imgs = _static(n)
    imgs[2] = None
    rec = compute_motion_sequence(_frames(n), imgs)
    assert rec["gaps"], "expected at least one gap"
    for g in rec["gaps"]:
        fr = g["frame_range"]
        assert fr["domain"] == "frame_no"
        assert fr["start"] < fr["end"]            # half-open needs start<end
        assert g["reason"] in ("missing_pts", "measurement_unavailable")
