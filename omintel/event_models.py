"""L1.5 Commit D -- frozen data contracts for in-shot event extraction.

This module defines ONLY the event data contract (models, enums, schema
version, validators and JSON round-trips). No detection algorithm lives
here; measurement code arrives in Commit E (``event_extractor.py``) and
the regression corpus in Commit F.

Architectural invariants (each mirrors an existing L0/L0.5 decision and
is enforced by :func:`validate_event_document` / :func:`validate_shot_span`):

1. **Time is PTS-authoritative.** Event times are ``pts_s`` values on the
   canonical L0 timeline (``frames.csv`` produced by
   ``omintel.media_facts``, source ``ffprobe_best_effort_timestamp_time``).
   Time is never derived from ``frame_no / fps``: the source may be VFR or
   carry non-monotonic PTS.  ``frame_no`` is a lookup key only.

2. **Half-open shot spans.** A shot owns frames
   ``[start_frame_no, end_frame_no)`` -- the boundary frame belongs to the
   *next* shot.  Likewise every event interval is half-open in both time
   (``[start_pts_s, end_pts_s)``) and frame space.  This makes adjacency
   unambiguous and prevents double-counting cut frames.

3. **Boundary support is evidence, not a label.** Whether an interval edge
   is backed by a reviewed hard cut is recorded structurally in
   :class:`BoundarySupport` (review status + detector agreement + boundary
   id), never inferred from a single free-form tag.  The cut-review
   statuses (CONFIRMED / REVIEW_REQUIRED / UNCONFIRMED) belong to *boundaries*,
   not to events; events carry their own independent lifecycle status.

4. **Uncertainty is explicit.** Missing measurements, lost track coverage
   and unresolvable frames are recorded as :class:`CoverageGap` entries.
   ``confidence`` is ``None`` until calibrated against a labeled test set
   (same rule as soft-constraint weights in
   ``omintel.object_constraints.schemas``); algorithms emit raw
   measurements instead of fabricated precision.

5. **Index-only evidence.** Evidence references sidecar artifacts
   (manifest rows, frame files, CSV slices) by path/key; heavy per-frame
   arrays are never inlined into the event JSON.

6. **Separation from L1 boundary candidates.** ``events.json`` written by
   ``omintel.fusion`` is a *shot-boundary candidate* artifact
   (``openmontage.evidence_pack.v1``).  In-shot events form their own
   document (``openmontage.shot_events.v1``) and must never be merged into
   that file.

Frame identity across layers follows the L0.5 association policy:
``frame_no`` + ``sha256``; PTS comes from the manifest field
``source_pts_time`` (never substituted by decoder wall-clock time), and
manifest frame paths are POSIX-style, relative to the decode ``out_dir``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Schema / version constants
# --------------------------------------------------------------------------

EVENTS_SCHEMA_VERSION = "openmontage.shot_events.v1"
ALGORITHM_VERSION = "l1.5-events-d-0.1.0"

#: Canonical PTS source inherited from L0 media facts (see omintel.media_facts).
CANONICAL_TIME_SOURCE = "ffprobe_best_effort_timestamp_time"

#: Frame identity association with L0.5 manifests: index key + content hash.
FRAME_IDENTITY_KEYS = ("frame_no", "sha256")

#: PTS field carried by L0.5 ``frame_manifest.jsonl`` rows.  Decoder-side
#: wall-clock timestamps must never substitute it.
MANIFEST_PTS_FIELD = "source_pts_time"

#: L0.5 manifest frame paths are POSIX-style and relative to decode out_dir.
MANIFEST_PATH_STYLE = "posix_relative_to_out_dir"


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

#: First-version event types.  ``object_entry_exit`` is contract-only in
#: Commit D: without trustworthy detection/tracking evidence the extractor
#: must refuse to emit it as anything but ``NEEDS_EVIDENCE`` (see rules).
EVENT_TYPES = ("peak_motion", "brightness_jump", "camera_acceleration", "object_entry_exit")

#: Independent event lifecycle.  Deliberately NOT the cut-review enum.
EVENT_STATUSES = ("CANDIDATE", "MEASURED", "NEEDS_EVIDENCE", "REJECTED")

#: Review status of a *shot boundary* (input evidence to the event layer).
#: These describe cuts, not events; they arrive via BoundarySupport.
BOUNDARY_REVIEW_STATUSES = ("CONFIRMED", "REVIEW_REQUIRED", "UNCONFIRMED")

#: Quality gate shared with the object-constraints layer convention.
QUALITY_GATE_STATUSES = ("PASS", "REPAIR", "REVIEW")

#: How a boundary edge got its support value.
BOUNDARY_SUPPORT_LEVELS = ("agreement", "single_detector", "video_edge", "manual_review", "none")

#: Which upstream layer produced a piece of evidence.
EVIDENCE_SOURCES = ("l0_media_facts", "l0_5_frame_manifest", "l1_shot_candidates", "l1_5_measurement")

#: Reason codes for explicitly recorded uncertainty (no silent gaps).
GAP_REASONS = (
    "missing_pts",            # frame row exists but PTS is null
    "decode_misalignment",    # L0.5 alignment status not one-to-one
    "measurement_unavailable",# algorithm could not compute this metric here
    "track_lost",             # object coverage interrupted (entry/exit class)
    "boundary_ambiguity",     # edges too close to resolve half-open ownership
)


class ContractError(ValueError):
    """Raised when an event document violates an architectural invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


# --------------------------------------------------------------------------
# Data models (dataclasses match project style: plain dicts at the edges)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    """Half-open interval [start, end) in one domain.

    ``domain`` is ``"frame_no"`` (integer indices into the canonical
    frames.csv) or ``"pts_s"`` (seconds on the canonical PTS timeline).
    Both endpoints are required together; a zero-length interval is valid
    only for point-like peaks after expansion (validators enforce
    start <= end, strict < is NOT required so a peak can collapse onto a
    single frame while still owning exactly that frame).
    """
    domain: str
    start: float
    end: float

    def contains(self, value: float) -> bool:
        """Half-open membership: start <= value < end."""
        return self.start <= value < self.end


@dataclass(frozen=True)
class BoundarySupport:
    """Evidence record for one shot-boundary edge.

    Captures *why* an edge may or may not be trusted before in-shot metrics
    near it are attributed to motion rather than to a cut.  ``review_status``
    is the human/cut-audit verdict (CONFIRMED / ...), which is orthogonal to
    detector agreement recorded in ``support_level``.
    """
    boundary_id: str
    review_status: str          # BOUNDARY_REVIEW_STATUSES
    support_level: str          # BOUNDARY_SUPPORT_LEVELS
    support_count: int          # number of detectors agreeing (0 allowed)
    pts_s: float | None         # edge time on canonical timeline
    frame_no: int | None        # first frame owned by the NEXT shot
    note: str | None = None

    def is_hard_cut(self) -> bool:
        """True only when a human CONFIRMED verdict backs the edge.

        Detector agreement alone never counts as confirmed -- a cut frame
        must be excluded from intra-shot attribution only on reviewed proof.
        """
        return self.review_status == "CONFIRMED"


@dataclass(frozen=True)
class ShotSpan:
    """A shot as consumed by L1.5: half-open frame ownership + edge evidence."""
    shot_id: str
    start_frame_no: int
    end_frame_no: int           # exclusive: frame belongs to next shot
    start_pts_s: float
    end_pts_s: float
    start_boundary: BoundarySupport | None = None
    end_boundary: BoundarySupport | None = None
    provisional: bool = True    # inherits L1 provisional-partition flag

    def owns_frame(self, frame_no: int) -> bool:
        return Interval("frame_no", self.start_frame_no, self.end_frame_no).contains(frame_no)

    def owns_pts(self, pts_s: float) -> bool:
        return Interval("pts_s", self.start_pts_s, self.end_pts_s).contains(pts_s)


@dataclass(frozen=True)
class FrameRef:
    """Reference to one decoded frame through the L0.5 manifest.

    Identity = frame_no + sha256 (association policy); time = source_pts_time;
    path = posix relative to decode out_dir.  ``alignment_status`` carries the
    L0.5 one-to-one audit verdict for this row.
    """
    frame_no: int
    source_pts_time: float | None
    path: str                   # posix, relative to out_dir
    sha256: str | None = None   # None => identity incomplete, must gap-log
    alignment_status: str | None = None


@dataclass(frozen=True)
class CoverageGap:
    """Explicit record of uncertainty; replaces silent interpolation."""
    reason: str                 # GAP_REASONS
    frame_range: Interval       # half-open, domain="frame_no"
    detail: str | None = None


@dataclass(frozen=True)
class AlgorithmProvenance:
    """Where a measurement came from, so results are re-runnable."""
    name: str
    algorithm_version: str
    params: dict[str, Any] = field(default_factory=dict)
    input_artifacts: dict[str, str] = field(default_factory=dict)  # role -> path


@dataclass(frozen=True)
class EventEvidence:
    """Index-only evidence bundle for one event.

    ``measurements`` holds raw algorithm outputs (e.g. mean abs diff,
    luminance delta) keyed by metric name -- never a smoothed scalar
    dressed up as probability.  ``frame_refs`` lists the associated frames;
    frames lacking identity/time must appear in ``gaps`` instead.
    """
    source: str                     # EVIDENCE_SOURCES
    frame_refs: list[FrameRef] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)  # comparison window stats
    gaps: list[CoverageGap] = field(default_factory=list)
    boundary_context: dict[str, BoundarySupport | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    """One in-shot event under ``openmontage.shot_events.v1``.

    ``confidence`` is Optional and defaults to None: it may only become a
    number once calibrated against a labeled benchmark (Commit F scope).
    Until then consumers must treat status + evidence, not any scalar, as
    the trust signal.
    """
    event_id: str
    shot_id: str
    event_type: str               # EVENT_TYPES
    status: str                   # EVENT_STATUSES
    interval_frames: Interval     # half-open [start_fn, end_fn)
    interval_pts: Interval        # half-open [start_pts, end_pts)
    peak_pts_s: float | None = None
    confidence: float | None = None
    confidence_note: str | None = None
    evidence: EventEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_interval(iv: Interval | dict[str, Any], *, within: tuple[float, float] | None = None,
                      label: str = "interval") -> None:
    d = iv.domain if isinstance(iv, Interval) else iv.get("domain")
    s = iv.start if isinstance(iv, Interval) else iv.get("start")
    e = iv.end if isinstance(iv, Interval) else iv.get("end")
    _require(d in ("frame_no", "pts_s"), f"{label}.domain must be 'frame_no' or 'pts_s', got {d!r}")
    _require(s is not None and e is not None, f"{label} requires both start and end (explicit, no defaults)")
    _require(float(s) <= float(e), f"{label} must satisfy start <= end (half-open [start, end))")
    if within is not None:
        lo, hi = within
        _require(float(s) >= float(lo), f"{label}.start {s} outside enclosing range [{lo}, {hi})")
        _require(float(e) <= float(hi), f"{label}.end {e} outside enclosing range [{lo}, {hi})")


def validate_shot_span(span: ShotSpan | dict[str, Any]) -> None:
    g = (lambda k: getattr(span, k)) if isinstance(span, ShotSpan) else (lambda k: span.get(k))
    _require(bool(g("shot_id")), "shot_id required")
    sfn, efn = g("start_frame_no"), g("end_frame_no")
    _require(isinstance(sfn, int) and isinstance(efn, int),
             "shot span needs integer frame_no bounds")
    _require(sfn < efn,
             "shot span must be a non-empty half-open range [start_frame_no, end_frame_no)")
    sp, ep = g("start_pts_s"), g("end_pts_s")
    _require(sp is not None and ep is not None,
             "shot span needs PTS endpoints from the canonical timeline")
    _require(float(sp) <= float(ep), "shot span start_pts_s must be <= end_pts_s")
    for side in ("start_boundary", "end_boundary"):
        b = g(side)
        if b is not None:
            bd = b if isinstance(b, dict) else asdict(b)
            _require(bd.get("review_status") in BOUNDARY_REVIEW_STATUSES,
                     f"{side}.review_status must be one of {BOUNDARY_REVIEW_STATUSES}")
            _require(bd.get("support_level") in BOUNDARY_SUPPORT_LEVELS,
                     f"{side}.support_level must be one of {BOUNDARY_SUPPORT_LEVELS}")


def validate_event(event: Event | dict[str, Any], shot: ShotSpan | dict[str, Any] | None = None) -> None:
    """Validate one event record against the frozen Commit-D contract."""
    e = event if isinstance(event, dict) else event.to_dict()

    _require(bool(e.get("event_id")), "event_id required")
    _require(bool(e.get("shot_id")), "shot_id required")
    _require(e.get("event_type") in EVENT_TYPES,
             f"event_type must be one of {EVENT_TYPES}, got {e.get('event_type')!r}")
    _require(e.get("status") in EVENT_STATUSES,
             f"status must be one of {EVENT_STATUSES}, got {e.get('status')!r}")

    iv_f = e.get("interval_frames") or {}
    iv_p = e.get("interval_pts") or {}
    validate_interval(iv_f, label="interval_frames")
    validate_interval(iv_p, label="interval_pts")

    if shot is not None:
        sg = (lambda k: getattr(shot, k)) if isinstance(shot, ShotSpan) else (lambda k: shot.get(k))
        validate_interval(iv_f, within=(sg("start_frame_no"), sg("end_frame_no")),
                          label="interval_frames (vs shot)")
        validate_interval(iv_p, within=(sg("start_pts_s"), sg("end_pts_s")),
                          label="interval_pts (vs shot)")

    conf = e.get("confidence")
    _require(conf is None or isinstance(conf, (int, float)),
             "confidence must be null (uncalibrated) or numeric; never a placeholder string")
    if conf is not None:
        _require(bool(e.get("confidence_note")),
                 "a numeric confidence requires confidence_note naming the calibration basis")

    # Rule: object_entry_exit cannot be asserted without tracking evidence.
    if e.get("event_type") == "object_entry_exit":
        _require(e.get("status") == "NEEDS_EVIDENCE",
                 "object_entry_exit is interface-only until detection/tracking "
                 "evidence exists; it must not be emitted as MEASURED/CANDIDATE")

    ev = e.get("evidence")
    if ev is not None:
        _require(ev.get("source") in EVIDENCE_SOURCES,
                 f"evidence.source must be one of {EVIDENCE_SOURCES}")
        # index-only invariant: no inlined per-frame arrays
        for forbidden in ("mask_array", "flow_array", "pixels"):
            _require(forbidden not in (ev.get("measurements") or {}),
                     f"evidence.measurements.{forbidden} must live in a sidecar file")
        for gap in ev.get("gaps") or []:
            gg = gap if isinstance(gap, dict) else asdict(gap)
            _require(gg.get("reason") in GAP_REASONS,
                     f"coverage gap reason must be one of {GAP_REASONS}")
        for fr in ev.get("frame_refs") or []:
            _require(isinstance(fr.get("frame_no"), int),
                     "every frame_ref needs an integer frame_no (L0 lookup key)")
            _require(MANIFEST_PTS_FIELD in fr,
                     f"frame_ref must carry '{MANIFEST_PTS_FIELD}' explicitly (null allowed, absence is not)")
            # Uncertainty invariant: a frame row without PTS (or without the
            # sha256 half of its identity) may only exist when an explicit
            # CoverageGap names that frame range.  No silent holes.
            if fr.get(MANIFEST_PTS_FIELD) is None or fr.get("sha256") is None:
                fn = fr["frame_no"]
                covered = any(
                    (gap if isinstance(gap, dict) else asdict(gap)).get("frame_range", {}) and
                    float((gap if isinstance(gap, dict) else asdict(gap))["frame_range"].get("start", 0)) <= fn
                    < float((gap if isinstance(gap, dict) else asdict(gap))["frame_range"].get("end", 0))
                    for gap in (ev.get("gaps") or []))
                _require(covered,
                         f"frame_no {fn} lacks PTS/sha256 and has no covering CoverageGap; "
                         "uncertainty must be recorded explicitly")


def validate_events_document(doc: dict[str, Any]) -> None:
    """Validate a whole ``openmontage.shot_events.v1`` document."""
    _require(doc.get("schema_version") == EVENTS_SCHEMA_VERSION,
             f"schema_version must be {EVENTS_SCHEMA_VERSION}")
    _require(bool(doc.get("video_id")) and bool(doc.get("source_sha256")),
             "video_id and source_sha256 are required (frame identity joins on frame_no+sha256)")
    tb = doc.get("time_base") or {}
    _require(tb.get("canonical_time_source") == CANONICAL_TIME_SOURCE,
             f"time_base.canonical_time_source must be {CANONICAL_TIME_SOURCE}")
    _require(tb.get("path_style") == MANIFEST_PATH_STYLE,
             f"time_base.path_style must be {MANIFEST_PATH_STYLE} (L0.5 manifest convention)")
    _require(str(tb.get("interval_convention")) == "[start, end)",
             "time_base.interval_convention must state the half-open rule")

    shots = {s["shot_id"]: s for s in doc.get("shots") or []}
    for s in shots.values():
        validate_shot_span(s)
    seen_ids: set[str] = set()
    for ev in doc.get("events") or []:
        _require(ev.get("event_id") not in seen_ids, f"duplicate event_id {ev.get('event_id')}")
        seen_ids.add(ev.get("event_id"))
        _require(ev.get("shot_id") in shots,
                 f"event {ev.get('event_id')} references unknown shot {ev.get('shot_id')!r}")
        validate_event(ev, shot=shots[ev["shot_id"]])

    gate = doc.get("quality_gate") or {}
    _require(gate.get("status") in QUALITY_GATE_STATUSES,
             f"quality_gate.status must be one of {QUALITY_GATE_STATUSES}")
    if gate.get("status") != "PASS":
        _require(bool(gate.get("reason")),
                 "non-PASS quality_gate must carry an explicit reason")


# --------------------------------------------------------------------------
# Serialization helpers
# --------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _jsonable(asdict(obj))
    return str(obj)


def build_events_document(*, video_id: str, source_sha256: str,
                          shots: list[dict[str, Any]], events: list[dict[str, Any]],
                          algorithm: AlgorithmProvenance | None = None,
                          interpretation_note: str | None = None) -> dict[str, Any]:
    """Assemble the canonical top-level document (and self-validate it)."""
    doc = {
        "schema_version": EVENTS_SCHEMA_VERSION,
        "algorithm_version": (algorithm or AlgorithmProvenance(name="contract_only",
                                                               algorithm_version=ALGORITHM_VERSION)).algorithm_version,
        "video_id": video_id,
        "source_sha256": source_sha256,
        "time_base": {
            "canonical_time_source": CANONICAL_TIME_SOURCE,
            "interval_convention": "[start, end)",
            "path_style": MANIFEST_PATH_STYLE,
            "frame_identity": list(FRAME_IDENTITY_KEYS),
            "manifest_pts_field": MANIFEST_PTS_FIELD,
        },
        "shots": shots,
        "events": events,
        "quality_gate": {"status": "REVIEW",
                         "reason": "Commit D contract only: no measurements, no calibration"},
        "interpretation_note": interpretation_note
        or "In-shot events only. Shot-boundary candidates remain in openmontage.evidence_pack.v1.",
    }
    validate_events_document(doc)
    return doc


def write_events_document(path: Any, doc: dict[str, Any]) -> None:
    """Deterministic serialization: sorted keys, UTF-8, trailing newline."""
    text = json.dumps(_jsonable(doc), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
