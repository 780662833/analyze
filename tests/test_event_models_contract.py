"""Commit D contract tests for omintel.event_models (no algorithms tested).

These tests freeze the data contract only: schema identity, half-open
interval semantics, PTS authoritativeness, boundary-support evidence,
explicit uncertainty, and separation from the L1 boundary-candidate
``events.json``.  They must keep passing unchanged through Commits E/F.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omintel import event_models as em
from omintel.event_models import (
    BoundarySupport,
    ContractError,
    CoverageGap,
    Event,
    EventEvidence,
    FrameRef,
    Interval,
    ShotSpan,
    build_events_document,
    validate_event,
    validate_events_document,
    validate_interval,
    validate_shot_span,
    write_events_document,
)


def bs(bid="B0001", status="CONFIRMED", level="agreement", pts=2.0, fn=60):
    return BoundarySupport(boundary_id=bid, review_status=status, support_level=level,
                           support_count=2 if level == "agreement" else 1,
                           pts_s=pts, frame_no=fn)


def span(**kw):
    base = dict(shot_id="S0001", start_frame_no=0, end_frame_no=100,
                start_pts_s=0.0, end_pts_s=10.0,
                start_boundary=bs("video_start", "CONFIRMED", "video_edge", 0.0, 0),
                end_boundary=bs("B0001", "CONFIRMED", "agreement", 10.0, 100))
    base.update(kw)
    return ShotSpan(**base)


def fref(fn=10, pts=1.0, path="frames/f000010.png", sha="ab" * 32):
    return FrameRef(frame_no=fn, source_pts_time=pts, path=path, sha256=sha,
                    alignment_status="one_to_one")


def ev(event_type="peak_motion", status="MEASURED", conf=None, **kw):
    base = dict(event_id="EVT_0001", shot_id="S0001", event_type=event_type, status=status,
                interval_frames=Interval("frame_no", 10, 14),
                interval_pts=Interval("pts_s", 1.0, 1.4), peak_pts_s=1.2,
                confidence=conf,
                evidence=EventEvidence(source="l1_5_measurement",
                                       frame_refs=[fref(10), fref(11), fref(12)],
                                       measurements={"mean_abs_diff": 7.5},
                                       baseline={"median_mean_abs_diff": 0.8}))
    base.update(kw)
    return Event(**base)


# ---------------------------------------------------------------- schema identity

def test_schema_and_constants():
    assert em.EVENTS_SCHEMA_VERSION == "openmontage.shot_events.v1"
    assert em.CANONICAL_TIME_SOURCE == "ffprobe_best_effort_timestamp_time"
    assert em.FRAME_IDENTITY_KEYS == ("frame_no", "sha256")
    assert em.MANIFEST_PTS_FIELD == "source_pts_time"
    assert em.MANIFEST_PATH_STYLE == "posix_relative_to_out_dir"
    # event lifecycle is independent from cut-review statuses
    assert set(em.EVENT_STATUSES).isdisjoint(set(em.BOUNDARY_REVIEW_STATUSES))


def test_document_roundtrip_and_self_validation():
    doc = build_events_document(video_id="vid", source_sha256="cd" * 32,
                                shots=[span().__dict__ | {"start_boundary": bs().__dict__,
                                                           "end_boundary": bs().__dict__}],
                                events=[])
    assert doc["schema_version"] == em.EVENTS_SCHEMA_VERSION
    assert doc["time_base"]["interval_convention"] == "[start, end)"
    validate_events_document(doc)


# ---------------------------------------------------------------- intervals

def test_half_open_membership():
    iv = Interval("frame_no", 10, 14)
    assert iv.contains(10) and not iv.contains(14) and iv.contains(13)
    pts = Interval("pts_s", 1.0, 1.4)
    assert pts.contains(1.0) and not pts.contains(1.4)


def test_interval_rejects_inverted_bounds():
    with pytest.raises(ContractError):
        validate_interval(Interval("frame_no", 14, 10))
    with pytest.raises(ContractError):
        validate_interval({"domain": "pts_s", "start": None, "end": 1.0})


def test_event_interval_must_fit_shot_half_open_range():
    # end_frame_no == shot.end_frame_no is allowed (exclusive edge), +1 is not
    validate_event(ev(interval_frames=Interval("frame_no", 90, 100)), shot=span())
    with pytest.raises(ContractError, match="outside enclosing range"):
        validate_event(ev(interval_frames=Interval("frame_no", 90, 101)), shot=span())
    # a frame equal to the boundary belongs to the NEXT shot -> event starting
    # at the exclusive end is out of range when it exceeds it
    with pytest.raises(ContractError):
        validate_event(ev(interval_pts=Interval("pts_s", 1.0, 10.5)), shot=span())


def test_shot_span_requires_nonempty_half_open_range():
    with pytest.raises(ContractError, match="half-open"):
        validate_shot_span(span(start_frame_no=50, end_frame_no=50))
    with pytest.raises(ContractError):
        validate_shot_span({"shot_id": "S1", "start_frame_no": 0, "end_frame_no": 10,
                            "start_pts_s": None, "end_pts_s": 1.0})


def test_boundary_ownership_is_half_open():
    s1 = span(shot_id="S0001", start_frame_no=0, end_frame_no=100, end_pts_s=10.0)
    s2 = span(shot_id="S0002", start_frame_no=100, end_frame_no=200,
              start_pts_s=10.0, end_pts_s=20.0)
    assert s1.owns_frame(99) and not s1.owns_frame(100)   # boundary frame -> next shot
    assert s2.owns_frame(100) and not s2.owns_frame(200)
    assert s1.owns_pts(9.999) and not s1.owns_pts(10.0)
    assert s2.owns_pts(10.0)


# ---------------------------------------------------------------- PTS authority

def test_frame_ref_requires_explicit_source_pts_field():
    e = ev()
    d = e.to_dict()
    d["evidence"]["frame_refs"][0].pop("source_pts_time")
    with pytest.raises(ContractError, match="source_pts_time"):
        validate_event(d)
    # null PTS is representable ONLY when an explicit gap covers the frame;
    # plain absence of the key is a structural violation either way.
    d["evidence"]["frame_refs"][0]["source_pts_time"] = None
    with pytest.raises(ContractError, match="CoverageGap"):
        validate_event(d)  # null PTS without a covering gap -> rejected


def test_missing_pts_requires_explicit_gap_reason():
    bad = ev(evidence=EventEvidence(
        source="l1_5_measurement",
        frame_refs=[FrameRef(frame_no=10, source_pts_time=None, path="frames/f10.png",
                             sha256="ab" * 32)],
        measurements={}, gaps=[]))
    with pytest.raises(ContractError, match="CoverageGap"):
        validate_event(bad)
    good = ev(evidence=EventEvidence(
        source="l1_5_measurement",
        frame_refs=[FrameRef(frame_no=10, source_pts_time=None, path="frames/f10.png",
                             sha256="ab" * 32)],
        measurements={},
        gaps=[CoverageGap(reason="missing_pts", frame_range=Interval("frame_no", 10, 11))]))
    validate_event(good)
    with pytest.raises(ContractError):
        validate_event(ev(evidence=EventEvidence(
            source="l1_5_measurement", frame_refs=[], measurements={},
            gaps=[CoverageGap(reason="guessed", frame_range=Interval("frame_no", 10, 11))])))


def test_null_sha256_also_requires_covering_gap():
    # identity is frame_no + sha256; a manifest row whose hash could not be
    # computed must be gap-logged, never silently trusted
    e = ev(evidence=EventEvidence(
        source="l0_5_frame_manifest",
        frame_refs=[FrameRef(frame_no=11, source_pts_time=1.1, path="frames/f11.png",
                             sha256=None, alignment_status="one_to_one")],
        measurements={}))
    with pytest.raises(ContractError, match="sha256|CoverageGap"):
        validate_event(e)
    ok = ev(evidence=EventEvidence(
        source="l0_5_frame_manifest",
        frame_refs=[FrameRef(frame_no=11, source_pts_time=1.1, path="frames/f11.png",
                             sha256=None, alignment_status="one_to_one")],
        measurements={},
        gaps=[CoverageGap(reason="decode_misalignment", frame_range=Interval("frame_no", 11, 12))]))
    validate_event(ok)


# ---------------------------------------------------------------- boundary evidence

def test_boundary_support_is_structural_not_a_bare_label():
    b = bs(status="REVIEW_REQUIRED", level="single_detector")
    assert not b.is_hard_cut()
    assert bs(status="CONFIRMED").is_hard_cut()
    # detector agreement WITHOUT human confirmation is not a hard cut
    assert not bs(status="UNCONFIRMED", level="agreement").is_hard_cut()
    with pytest.raises(ContractError):
        validate_shot_span(span(end_boundary={"boundary_id": "x", "review_status": "AGREED",
                                              "support_level": "agreement", "support_count": 2,
                                              "pts_s": 1.0, "frame_no": 100}))


def test_confirmed_hard_cut_excludes_boundary_frame_from_left_shot():
    # The regression rule for acceptance criterion 3: a剧烈 change AT the
    # confirmed cut frame cannot be attributed to the left shot, because the
    # frame belongs to the right shot under [start, end).
    s1, s2 = span(), span(shot_id="S0002", start_frame_no=100, end_frame_no=200,
                          start_pts_s=10.0, end_pts_s=20.0)
    assert s1.end_boundary.is_hard_cut() and s2.start_boundary.is_hard_cut()
    cut_frame = 100
    assert not s1.owns_frame(cut_frame)
    assert s2.owns_frame(cut_frame)
    with pytest.raises(ContractError):
        # an event in S0001 that swallows the cut frame is invalid
        validate_event(ev(shot_id="S0001", interval_frames=Interval("frame_no", 96, 101)),
                       shot=s1)


# ---------------------------------------------------------------- uncertainty / confidence

def test_confidence_defaults_to_none_and_needs_calibration_note():
    assert ev().confidence is None
    validate_event(ev())  # uncalibrated null confidence is valid
    with pytest.raises(ContractError, match="calibration"):
        validate_event(ev(conf=0.9))  # numeric without stated basis -> rejected
    validate_event(ev(conf=0.9, confidence_note="calibrated on labeled set X (Commit F)"))
    with pytest.raises(ContractError):
        validate_event(ev(conf="high"))  # never a placeholder string


def test_object_entry_exit_cannot_be_asserted_without_tracking_evidence():
    with pytest.raises(ContractError, match="object_entry_exit"):
        validate_event(ev(event_type="object_entry_exit", status="MEASURED"))
    with pytest.raises(ContractError, match="object_entry_exit"):
        validate_event(ev(event_type="object_entry_exit", status="CANDIDATE"))
    ok = ev(event_type="object_entry_exit", status="NEEDS_EVIDENCE",
            evidence=EventEvidence(source="l1_5_measurement", frame_refs=[],
                                    measurements={},
                                    gaps=[CoverageGap(reason="track_lost",
                                                      frame_range=Interval("frame_no", 0, 100),
                                                      detail="no tracker backend in Commit D")]))
    validate_event(ok)


def test_unknown_enums_rejected():
    with pytest.raises(ContractError):
        validate_event(ev(event_type="camera_shake"))
    with pytest.raises(ContractError):
        validate_event(ev(status="REVIEW_REQUIRED"))  # cut status != event status
    with pytest.raises(ContractError):
        validate_event(ev(evidence=EventEvidence(source="llm_vibes")))


def test_index_only_evidence_rule():
    bad = ev(evidence=EventEvidence(source="l1_5_measurement",
                                    measurements={"flow_array": [[0.0]]}))
    with pytest.raises(ContractError, match="sidecar"):
        validate_event(bad)


# ---------------------------------------------------------------- document level

def _doc(shots, events):
    return build_events_document(video_id="vid", source_sha256="cd" * 32,
                                 shots=[shots], events=[events])


def test_document_rejects_duplicate_event_ids_and_orphan_shot_refs():
    s = span().__dict__ | {"start_boundary": bs().__dict__, "end_boundary": bs().__dict__}
    e = ev().to_dict()
    _doc(s, e)  # valid
    # duplicate event_id: two events with the same id inside one document
    e2 = dict(e, interval_frames={"domain": "frame_no", "start": 20, "end": 24},
              interval_pts={"domain": "pts_s", "start": 2.0, "end": 2.4})
    doc = {"schema_version": em.EVENTS_SCHEMA_VERSION, "video_id": "v",
           "source_sha256": "cd" * 32,
           "time_base": {"canonical_time_source": em.CANONICAL_TIME_SOURCE,
                         "path_style": em.MANIFEST_PATH_STYLE,
                         "interval_convention": "[start, end)"},
           "shots": [s], "events": [e, e2],
           "quality_gate": {"status": "REVIEW", "reason": "test"}}
    with pytest.raises(ContractError, match="duplicate"):
        validate_events_document(doc)
    e3 = dict(e, shot_id="S9999")
    doc["events"] = [e3]
    with pytest.raises(ContractError, match="unknown shot"):
        validate_events_document(doc)


def test_quality_gate_requires_reason_when_not_pass():
    s = span().__dict__ | {"start_boundary": bs().__dict__, "end_boundary": bs().__dict__}
    doc = build_events_document(video_id="v", source_sha256="cd" * 32, shots=[s], events=[])
    doc["quality_gate"] = {"status": "PASS"}
    validate_events_document(doc)
    doc["quality_gate"] = {"status": "REVIEW"}  # missing reason
    with pytest.raises(ContractError, match="reason"):
        validate_events_document(doc)


def test_serialization_is_deterministic(tmp_path):
    s = span().__dict__ | {"start_boundary": bs().__dict__, "end_boundary": bs().__dict__}
    doc = build_events_document(video_id="v", source_sha256="cd" * 32, shots=[s],
                                events=[ev().to_dict()])
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    write_events_document(p1, doc)
    write_events_document(p2, doc)
    assert p1.read_bytes() == p2.read_bytes()
    loaded = json.loads(p1.read_text(encoding="utf-8"))
    validate_events_document(loaded)
    # sorted keys => byte-stable across rebuilds
    text = p1.read_text(encoding="utf-8")
    assert text.index('"algorithm_version"') < text.index('"events"')


def test_separation_from_l1_boundary_events_json():
    # L1 fusion writes openmontage.evidence_pack.v1 to events.json;
    # the event contract uses its own schema version and filename convention.
    assert em.EVENTS_SCHEMA_VERSION == "openmontage.shot_events.v1"
    assert em.EVENTS_SCHEMA_VERSION != "openmontage.evidence_pack.v1"
