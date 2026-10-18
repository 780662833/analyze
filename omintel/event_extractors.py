"""L1.5 Commit E (E1b): brightness_jump candidate generation.

Layering honored: E1 produced the raw measurement sequence
(``event_measurements.compute_luminance_sequence``); THIS module only applies
explicit, auditable criteria to that record.  It never re-decodes video and
never recomputes luminance statistics.

Semantics (reviewed design): a ``brightness_jump`` here is a *significant
whole-frame luminance-change candidate*, NOT an assertion that lighting
changed.  Whole-frame mean luma cannot distinguish illumination change,
exposure change, flash effects, or content突变; classification beyond
"candidate" requires additional evidence (Commit F scope).

Contract rules honored (event_models.py, Commit D):
  * Events live inside one shot's half-open frame ownership [start, end);
    a transition whose post-cut frame belongs to the NEXT shot is attributed
    to that shot -- never split across two.
  * PTS authority: event times come from the measurement record's pts_s
    (verbatim L0 values), never frame_no / fps.
  * Boundary evidence: a CONFIRMED hard cut adjacent to a large delta marks
    the candidate boundary-attributed (suppressed as intra-shot claim, kept
    in ``boundary_attributed_transitions`` for audit).  Detector-agreement
    or single-path boundaries are NEVER treated as confirmed: candidates near
    them stay CANDIDATE with boundary context recorded.
  * Uncertainty: frames missing PTS/identity, unreadable frames, and
    insufficient coverage become explicit CoverageGap entries; no effective
    candidate is emitted without usable PTS on both sides of the delta.
  * Determinism: pure function of (measurement record, shots, params,
    ALGORITHM_VERSION); stable iteration order; floats rounded at one point.
  * No LLM, no model downloads, no GPU.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from omintel.event_models import (
    AlgorithmProvenance,
    BoundarySupport,
    ContractError,
    Event,
    EventEvidence,
    FrameRef,
    Interval,
    ShotSpan,
    build_events_document,
    validate_event,
    _jsonable,
)

#: bump when candidate semantics change; recorded in provenance + document.
ALGORITHM_NAME = "brightness_jump_candidates"
ALGORITHM_VERSION = "1.0.0"

#: Experiment defaults ONLY -- must be overridden per material via params;
#: they are echoed into AlgorithmProvenance.params so no magic number is
#: silently baked into results.
DEFAULT_PARAMS: dict[str, Any] = {
    # minimum |delta_mean_luma| (8-bit luma units) to qualify as a jump
    "delta_threshold": 8.0,
    # fixed window (frames) used for local baseline stats around a candidate
    "window_frames": 5,
    # max gap between consecutive qualifying deltas merged into one event
    "merge_gap_frames": 1,
    # measurement coverage_ratio required before any candidate is trusted
    "min_coverage_ratio": 0.9,
}


def _r(v: float, precision: int = 6) -> float:
    return float(round(float(v), precision))


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return _r(s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0)


def _frame_refs(m: dict[str, Any], idxs: list[int],
                sha_by_fn: dict[int, str] | None) -> tuple[list[FrameRef], list[dict[str, Any]]]:
    """Build index-only frame references + auto gaps for incomplete identity.

    Contract invariant (event_models.validate_event): a frame_ref without PTS
    or without the sha256 half of its identity may only exist when an
    explicit CoverageGap names that frame range.  Rather than let extraction
    crash -- or worse, silently drop the ref -- we emit the gap here with a
    precise reason: missing PTS -> ``missing_pts``; PTS present but no
    manifest hash -> ``decode_misalignment`` (identity cannot be joined to
    the L0.5 manifest).
    """
    refs: list[FrameRef] = []
    gaps: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i in idxs:
        fn = int(m["frame_nos"][i])
        if fn in seen:
            continue
        seen.add(fn)
        pts = m["pts_s"][i]
        sha = (sha_by_fn or {}).get(fn)
        if pts is None:
            gaps.append({"reason": "missing_pts",
                         "frame_range": {"domain": "frame_no", "start": fn, "end": fn + 1},
                         "detail": "referenced frame has no usable L0 PTS"})
        elif sha is None:
            gaps.append({"reason": "decode_misalignment",
                         "frame_range": {"domain": "frame_no", "start": fn, "end": fn + 1},
                         "detail": "no L0.5 manifest sha256 supplied for this frame; "
                                   "frame_no+sha256 identity join incomplete"})
        refs.append(FrameRef(
            frame_no=fn,
            source_pts_time=(None if pts is None else _r(pts)),
            path=f"frames/{fn:06d}.png",   # manifest convention placeholder; real path comes from L0.5 manifest rows when provided
            sha256=sha,
            alignment_status="one_to_one" if sha else None,
        ))
    return refs, gaps


def _shot_span_dict(sh: ShotSpan) -> dict[str, Any]:
    """Serialize a ShotSpan for the document with EXACT dataclass key sets.

    The frozen Commit-D validator checks boundary dicts by key membership
    (asdict(BoundarySupport) == {boundary_id, review_status, support_level,
    support_count, pts_s, frame_no, note}); _jsonable's str() fallback would
    otherwise flatten nested dataclasses into strings and fail validation.
    asdict keeps the nested shapes correct; _jsonable then only normalizes
    scalars/lists/dicts.
    """
    return _jsonable(asdict(sh))


def find_brightness_candidates(
    measurement: dict[str, Any],
    *,
    params: dict[str, Any] | None = None,
    sha_by_fn: dict[int, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply initial screening + local baseline stats to a luminance record.

    Returns (candidates, analysis) where each candidate is a plain dict:
      pair_index, pre_frame_no, post_frame_no, pre_pts_s, post_pts_s,
      delta_mean_luma, local_median_abs_delta, local_max_abs_delta,
      window {start,end}, valid_samples, boundary_ambiguous (bool).
    ``analysis`` carries aggregates used later (low_coverage flag, gaps...).
    A candidate is *screening-level*; shot attribution happens downstream.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    thr = float(p["delta_threshold"])
    win = int(p["window_frames"])
    min_cov = float(p["min_coverage_ratio"])

    deltas = measurement["delta_mean_luma"]
    frame_nos = measurement["frame_nos"]
    pts_s = measurement["pts_s"]
    coverage = float(measurement.get("coverage_ratio", 0.0))

    abs_d = [None if d is None else abs(d) for d in deltas]
    cand: list[dict[str, Any]] = []
    for i, d in enumerate(deltas):
        if d is None or abs(d) < thr:
            continue
        lo, hi = max(0, i - win), min(len(deltas), i + win + 1)
        neigh = [a for j, a in enumerate(abs_d[lo:hi], start=lo)
                 if a is not None and j != i]
        pre_pts, post_pts = pts_s[i], pts_s[i + 1]
        # Uncertainty invariant: no candidate without PTS on both sides.
        boundary_ambiguous = pre_pts is None or post_pts is None
        cand.append({
            "pair_index": i,
            "pre_frame_no": int(frame_nos[i]),
            "post_frame_no": int(frame_nos[i + 1]),
            "pre_pts_s": None if pre_pts is None else _r(pre_pts),
            "post_pts_s": None if post_pts is None else _r(post_pts),
            "delta_mean_luma": _r(d),
            "local_median_abs_delta": _median(neigh),
            "local_max_abs_delta": (None if not neigh else _r(max(neigh))),
            "window": {"start": lo, "end": hi},
            "valid_samples": len(neigh),
            "boundary_ambiguous": boundary_ambiguous,
        })

    analysis = {
        "threshold_applied": _r(thr),
        "window_frames": win,
        "coverage_ratio": _r(coverage),
        "low_coverage": coverage < min_cov,
        "n_deltas": len(deltas),
        "n_valid_deltas": sum(1 for d in deltas if d is not None),
        "params": p,
    }
    return cand, analysis


def merge_runs(candidates: list[dict[str, Any]],
               measurement: dict[str, Any],
               threshold: float,
               merge_gap_frames: int) -> list[list[dict[str, Any]]]:
    """Deterministically group consecutive qualifying deltas into runs.

    Rules (all deterministic, all auditable):
      * adjacent qualifying deltas (pair index differs by 1) always merge --
        one uninterrupted excursion (e.g. a multi-frame flash rise);
      * a separation of up to ``merge_gap_frames`` pairs merges only when
        EVERY intervening delta is valid and at least half the threshold --
        a mild valley inside one excursion must not fragment it;
      * a None delta or a deep valley splits runs, so "flash up" and the
        later "flash down" across a stable plateau stay distinct events.
    """
    if not candidates:
        return []
    deltas = measurement["delta_mean_luma"]
    bridge_floor = float(threshold) / 2.0
    runs: list[list[dict[str, Any]]] = [[candidates[0]]]
    for c in candidates[1:]:
        prev = runs[-1][-1]
        gap = c["pair_index"] - prev["pair_index"]
        if gap == 1:
            runs[-1].append(c)
            continue
        ok_span = gap <= merge_gap_frames + 1
        bridged = all(deltas[j] is not None and abs(deltas[j]) >= bridge_floor
                      for j in range(prev["pair_index"] + 1, c["pair_index"]))
        if ok_span and bridged:
            runs[-1].append(c)
        else:
            runs.append([c])
    return runs


def extract_brightness_events(
    measurement: dict[str, Any],
    shots: list[ShotSpan],
    *,
    video_id: str,
    source_sha256: str,
    params: dict[str, Any] | None = None,
    sha_by_fn: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Attribute merged brightness runs to shots and emit the events document.

    Attribution rule (half-open ownership): a transition delta[i] (between
    frame_nos[i] and frame_nos[i+1]) belongs to the shot that owns the POST
    frame -- the cut frame itself belongs to the next shot, so a hard cut's
    luminance collapse is evaluated inside its new shot, and runs are split
    at shot boundaries rather than spanning them.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    cand, analysis = find_brightness_candidates(measurement, params=p, sha_by_fn=sha_by_fn)
    runs = merge_runs(cand, measurement, float(p["delta_threshold"]), int(p["merge_gap_frames"]))

    events: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    extra_gaps: list[dict[str, Any]] = []
    counter = 0

    def _shot_of(fn: int) -> ShotSpan | None:
        for s in shots:
            if s.owns_frame(fn):
                return s
        return None

    for run in runs:
        # split run at shot boundaries (events must not cross shot edges)
        segments: list[tuple[ShotSpan | None, list[dict[str, Any]]]] = []
        for c in run:
            sh = _shot_of(c["post_frame_no"])
            key = sh.shot_id if sh else None
            if segments and segments[-1][0] is not None and key == (segments[-1][0].shot_id if segments[-1][0] else None):
                segments[-1][1].append(c)
            elif segments and segments[-1][0] is None and key is None:
                segments[-1][1].append(c)
            else:
                segments.append((sh, [c]))

        for sh, seg in segments:
            first, last = seg[0], seg[-1]
            peak = max(seg, key=lambda c: abs(c["delta_mean_luma"]))
            boundary_ctx: dict[str, BoundarySupport | None] = {}
            is_hard_cut_edge = False
            if sh is not None:
                for side, bd in (("start_boundary", sh.start_boundary), ("end_boundary", sh.end_boundary)):
                    if bd is not None:
                        boundary_ctx[side] = bd
                # transition whose post frame IS the confirmed-cut frame
                if (sh.start_boundary is not None
                        and sh.start_boundary.is_hard_cut()
                        and first["post_frame_no"] == sh.start_frame_no):
                    is_hard_cut_edge = True
            else:
                boundary_ctx["shot"] = None

            base = {
                "run_pair_indices": [c["pair_index"] for c in seg],
                "delta_mean_luma": peak["delta_mean_luma"],
                "pre_frame_no": first["pre_frame_no"],
                "post_frame_no": last["post_frame_no"],
                "peak_frame_no": peak["post_frame_no"],
                "peak_pts_s": peak["post_pts_s"],
                "local_stats": {
                    "median_abs_delta": peak["local_median_abs_delta"],
                    "max_abs_delta": peak["local_max_abs_delta"],
                    "window": peak["window"],
                    "valid_samples": peak["valid_samples"],
                },
                "threshold_applied": analysis["threshold_applied"],
            }

            if sh is None:
                # unattributable frame range -> explicit gap, never a fake event
                extra_gaps.append({
                    "reason": "boundary_ambiguity",
                    "frame_range": {"domain": "frame_no",
                                     "start": first["pre_frame_no"],
                                     "end": last["post_frame_no"] + 1},
                    "detail": "transition frames fall outside every shot span; cannot attribute intra-shot",
                })
                continue

            # Half-open clip: the transition pair may straddle a shot edge
            # (pre frame belongs to the previous shot).  The event interval
            # must stay inside its owning shot [start, end); the pre-frame is
            # still recorded in measurements for audit.
            clipped_start = max(first["pre_frame_no"], sh.start_frame_no)
            clipped_end = min(last["post_frame_no"] + 1, sh.end_frame_no)
            fn_list = measurement["frame_nos"]
            i_lo = fn_list.index(clipped_start)
            i_hi = fn_list.index(clipped_end - 1)
            pts_vals = [measurement["pts_s"][i] for i in range(i_lo, i_hi + 1)]
            iv_pts_start = next((v for v in pts_vals if v is not None), None)
            iv_pts_end = next((v for v in reversed(pts_vals) if v is not None), None)
            if iv_pts_start is None or iv_pts_end is None:
                # no usable PTS anywhere in the clipped span -> cannot state
                # an authoritative time interval: record gap, emit nothing
                extra_gaps.append({
                    "reason": "missing_pts",
                    "frame_range": {"domain": "frame_no",
                                     "start": clipped_start, "end": clipped_end},
                    "detail": "candidate has no usable L0 PTS inside its owning shot; "
                              "time interval cannot be stated authoritatively",
                })
                continue

            if is_hard_cut_edge:
                # CONFIRMED cut: do not claim it as an intra-shot lighting
                # event; keep for audit with its boundary evidence attached.
                suppressed.append({**base,
                                   "shot_id": sh.shot_id,
                                   "reason": "adjacent_to_confirmed_hard_cut",
                                   "boundary_support": _jsonable(sh.start_boundary)})
                continue

            counter += 1
            status = "CANDIDATE"
            seg_gaps = list(measurement["gaps"])
            for c in seg:
                if c["boundary_ambiguous"]:
                    seg_gaps.append({
                        "reason": "missing_pts",
                        "frame_range": {"domain": "frame_no",
                                         "start": c["post_frame_no"],
                                         "end": c["post_frame_no"] + 1},
                        "detail": "candidate delta touches a PTS-less frame",
                    })
            ref_idxs = [i for i in ([c["pair_index"] for c in seg]
                                    + [seg[-1]["pair_index"] + 1])
                        if clipped_start <= int(measurement["frame_nos"][i]) < clipped_end]
            refs, ref_gaps = _frame_refs(measurement, ref_idxs, sha_by_fn)
            seg_gaps.extend(ref_gaps)
            ev = Event(
                event_id=f"EVT_{counter:04d}",
                shot_id=sh.shot_id,
                event_type="brightness_jump",
                status=status,
                interval_frames=Interval("frame_no", clipped_start, clipped_end),
                interval_pts=Interval("pts_s", _r(iv_pts_start), _r(iv_pts_end)),
                peak_pts_s=(peak["post_pts_s"]
                             if (peak["post_pts_s"] is not None
                                 and iv_pts_start <= peak["post_pts_s"] <= iv_pts_end)
                             else _r(iv_pts_start)),
                confidence=None,
                confidence_note=("calibration benchmark pending (Commit F); "
                                  "trust status+evidence, not a scalar"),
                evidence=EventEvidence(
                    source="l1_5_measurement",
                    frame_refs=refs,
                    measurements={**base, "coverage_ratio": analysis["coverage_ratio"]},
                    baseline={
                        "pre_window_mean_luma": _median([m for m in
                            [measurement["mean_luma"][i] for i in
                             range(max(0, first["pair_index"] - int(p["window_frames"])),
                                   first["pair_index"])] if m is not None]),
                        "post_window_mean_luma": _median([m for m in
                            [measurement["mean_luma"][i] for i in
                             range(last["pair_index"] + 1,
                                   min(len(measurement["mean_luma"]),
                                       last["pair_index"] + 1 + int(p["window_frames"])))] if m is not None]),
                    },
                    gaps=seg_gaps,
                    boundary_context=boundary_ctx,
                ),
            )
            validate_event(ev, shot=sh)
            events.append(ev.to_dict())

    prov = AlgorithmProvenance(name=ALGORITHM_NAME, algorithm_version=ALGORITHM_VERSION,
                                params=p, input_artifacts={"measurement_input_digest": measurement["input_digest"]})
    quality_reason = None
    if analysis["low_coverage"]:
        quality_reason = (f"measurement coverage_ratio {analysis['coverage_ratio']} "
                          f"< min {p['min_coverage_ratio']}; candidates withheld from PASS")
    gate = {"status": "REVIEW",
            "reason": quality_reason or "no calibration benchmark yet; PASS requires explicit quality conditions (Commit F)"}
    doc = build_events_document(video_id=video_id, source_sha256=source_sha256,
                                 shots=[_shot_span_dict(s) for s in shots], events=events,
                                 algorithm=prov)
    doc["quality_gate"] = gate
    doc["measurement_provenance"] = {
        "algorithm": measurement["algorithm"],
        "algorithm_version": measurement["algorithm_version"],
        "input_digest": measurement["input_digest"],
    }
    doc["boundary_attributed_transitions"] = suppressed
    doc["unattributed_transitions"] = extra_gaps
    return doc
