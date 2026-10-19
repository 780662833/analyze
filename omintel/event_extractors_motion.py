"""L1.5 Commit E (E2b): peak_motion candidate extraction over E2a measures.

Layering honored: E2a (``event_motion.compute_motion_sequence``) produced the
raw frame-pair picture-change sequence; THIS module only applies explicit,
auditable criteria to that record.  It never re-decodes video, never
recomputes pixel differences, performs no disk writes and never touches L2
events.json.

v1.0.0 criteria contract (frozen in review, revisions R1-R4 applied):

  Peak eligibility (R1, frozen ruling 1 -- SINGLE prominence formula):
    * absolute floor: mean_abs_diff >= abs_threshold (params, provenance);
    * local prominence (AUTHORITATIVE FORMULA, strict >): the plateau value
      must satisfy BOTH
          v > max(N) * prominence_ratio   AND   v > median(N)
      where N = the valid, UNSUPPRESSED pairs within +/- window_frames --
      the whole plateau is EXCLUDED from N.  The median clause compares
      against the RAW median (no ratio scaling).  A plateau whose equal
      neighbors reach its own height therefore LOSES prominence
      (v > v * r is false) -- that is intended behaviour, e.g. it keeps a
      constant-translation platform peak-free apart from the onset edge.
      ``prominence_max_ratio`` is DEPRECATED / INACTIVE: it does not
      participate in peak selection and is stripped from the ACTIVE
      parameter set echoed into provenance;
    * flat top: consecutive equal-valued qualifying pairs form ONE plateau
      whose representative index is start + (len-1)//2 (LEFT-MIDDLE for even
      lengths) -- deterministic regardless of traversal order;
    * edge peaks are allowed but carry ``single_sided_window`` evidence;
    * a pair adjacent to a CONFIRMED hard cut (BoundarySupport.is_hard_cut()
      -- the ONLY source of confirmed semantics, no new enums, R3) is
      peak-INELIGIBLE: it is recorded verbatim in
      ``audit.suppressed_peaks`` with its boundary evidence attached.

  Activity runs and merging (frozen ruling -- THREE STAGES, THREE FLOORS):
    * stage 1 peak eligibility   : absolute floor abs_threshold (+ the
      prominence formula above);
    * stage 2 run extension      : a run starts at each eligible peak's
      whole plateau and grows outward ONLY over ACTIVE pairs -- valid,
      frame-contiguous and >= ACTIVITY FLOOR
      (abs_threshold * activity_floor_ratio, DEFAULT 0.1; frozen rule,
      restored after review: index-space refactoring must NEVER change
      this floor).  A valid pair strictly below that floor is RESTING and
      stops growth; crossing resting stretches is governed by the
      deterministic bounded bridge (budget merge_gap_frames) implemented
      in ExcursionPlanner.extend().  Extension stops at invalid pairs
      (nothing spans missing data), at non-contiguous frame numbers (real
      frame_no holes) and at confirmed cut-crossing pairs (forced
      segmentation);
    * stage 3 run merging        : two peak-centered runs join into ONE
      candidate only when they overlap/touch or are separated by
      <= merge_gap_frames intervening pairs, EVERY one of which is valid
      and >= abs_threshold/2 (BRIDGE FLOOR, same spirit as E1b).
      The three floors are deliberately distinct: prominence_ratio scales
      LOCAL prominence only; abs_threshold/2 is the MERGE bridge floor;
      neither may be substituted for the other.  A deep-valley split test
      therefore uses a valley strictly BELOW abs_threshold/2 together with
      merge_gap_frames small enough to pin the bridge rule itself.

  Candidate interval (R2, post-frame semantics):
    * an ACTIVE pair measures a real change between its pre and post
      frames, so BOTH endpoints belong to the observed excursion: active
      pair range [a..b] maps to the half-open frame interval
      [pairs[a].frame_no, pairs[b].frame_no + 1) -- the FIRST post frame
      is the left anchor, the LAST post frame stays inside the interval
      (right-open).  Verified through the REAL frame numbers (never list
      positions +/- 1); prev_frame_no of the first pair is NOT part of
      this run's change evidence;
    * a sub-floor RESTING pair inside a merged burst contributes no change
      evidence of its own; its pre frame is already covered as the post
      frame of the preceding pair, so the mapping never loses a boundary
      frame and never fabricates one across a hole;
    * non-contiguous coverage NEVER collapses into a fake +1 span (the
      run segments instead);
    * attribution follows the existing ShotSpan half-open ownership and the
      E1b post-frame rule (a change belongs to the shot owning its post
      frame); validate_event enforces containment inside the shot.

  Time and uncertainty (R4):
    * all times come from the measurement record's authoritative PTS values
      (never fps arithmetic); if the strongest peak has no PTS the candidate
      is NOT emitted -- it lands in ``audit.candidates_without_peak_pts``
      with an explicit ``missing_pts`` gap;
    * unattributable ranges become explicit ``boundary_ambiguity`` gaps;
    * ambiguity near UNCONFIRMED boundaries is recorded as
      ``review_reasons`` / ``boundary_context``, never disguised as a
      CoverageGap when measurements are complete (R3);
    * brightness-transition signature (huge diff, zero changed-pixel ratio)
      is flagged via ``possible_brightness_transition`` -- retained as a
      candidate, classification deferred downstream;
    * every event stays CANDIDATE with confidence=None until calibration;
      quality_gate defaults to REVIEW with an explicit reason.

Determinism: pure function of (measurement record, shots, params,
ALGORITHM_VERSION); stable iteration order; floats rounded at one canonical
point; identical inputs serialize byte-identically.  No LLM, no GPU, no
network, no model downloads.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from omintel.event_models import (
    AlgorithmProvenance,
    ContractError,
    Event,
    EventEvidence,
    Interval,
    ShotSpan,
    build_events_document,
    validate_event,
    _jsonable,
)
from omintel.event_extractors import _frame_refs, _r, _shot_span_dict

#: bump when candidate semantics change; recorded in provenance + document.
ALGORITHM_NAME = "peak_motion_candidates"
ALGORITHM_VERSION = "1.0.0"

#: Experiment defaults ONLY -- must be overridden per material via params;
#: they are echoed into AlgorithmProvenance.params so no magic number is
#: silently baked into results.  ``prominence_max_ratio`` is the single
#: exception: DEPRECATED / INACTIVE (frozen ruling 1) and stripped from
#: every ACTIVE parameter set before it reaches provenance.
DEFAULT_PARAMS: dict[str, Any] = {
    # absolute floor on mean_abs_diff (uint8 units) for peak eligibility
    "abs_threshold": 5.0,
    # AUTHORITATIVE prominence formula (frozen ruling 1, strict >):
    #     plateau_value > max(neighbors) * prominence_ratio
    #   AND                     plateau_value > median(neighbors)
    # Neighbors exclude the whole plateau and all blocked pairs.  The
    # median clause compares against the RAW median -- prominence_ratio
    # scales the MAX clause only.  Equal-strength neighbours therefore
    # defeat prominence (v > v*r false), which is intended: a constant
    # translation platform stays peak-free apart from its onset edge.
    "prominence_ratio": 0.5,
    # DEPRECATED / INACTIVE (frozen ruling 1): retained only for parameter
    # compatibility; it does NOT participate in peak selection and must not
    # be echoed into active provenance params.
    "prominence_max_ratio": None,
    # fixed neighborhood window (pairs) used for prominence stats
    "window_frames": 3,
    # stage-2 ACTIVITY FLOOR ratio (frozen contract, restored after review):
    # a run grows outward over valid pairs >= abs_threshold * this ratio;
    # a valid pair strictly below it is RESTING and stops growth.  The
    # index-space refactoring must never silently change this floor.
    "activity_floor_ratio": 0.1,
    # max gap between peaks merged into one candidate (bridge rule, stage 3)
    "merge_gap_frames": 1,
    # measurement coverage_ratio required before any candidate is trusted
    "min_coverage_ratio": 0.9,
    # brightness-transition heuristic (frozen review ruling): a peak whose
    # changed_pixel_ratio is HIGH (>= brightness_like_ratio_min) AND whose
    # max_abs_diff stays within brightness_like_max_over_mean of its
    # mean_abs_diff (uniform whole-frame change, no spatial structure) is
    # flagged possible_brightness_transition -- a descriptive marker only,
    # never a semantic classification.
    "brightness_like_ratio_min": 0.9,
    "brightness_like_max_over_mean": 1.5,
}


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _pair_value(m: dict[str, Any], i: int) -> float | None:
    pr = m["frame_pairs"][i]
    return None if not pr["valid"] else float(pr["mean_abs_diff"])


def _confirmed_cut_frame_nos(shots: list[ShotSpan]) -> set[int]:
    """Frame numbers of edges that are CONFIRMED hard cuts -- exclusively via
    the frozen BoundarySupport.is_hard_cut() contract (R3: no new enums)."""
    out: set[int] = set()
    for s in shots:
        for bd in (s.start_boundary, s.end_boundary):
            if bd is not None and bd.is_hard_cut():
                out.add(int(bd.frame_no))
    return out


def _plateau_representatives(vals: list[float | None], elig: list[bool],
                             ) -> list[int]:
    """Collapse maximal plateaus of equal values among eligible indices.

    Consecutive eligible indices with identical values form one plateau;
    the representative is ``start + (length - 1) // 2`` -- the exact middle
    for odd lengths, the LEFT-MIDDLE for even lengths (frozen R1 rule).
    Returns representatives in ascending order.
    """
    reps: list[int] = []
    plateau: list[int] = []

    def flush() -> None:
        if plateau:
            reps.append(plateau[0] + (len(plateau) - 1) // 2)
            plateau.clear()

    prev_val = object()
    for i, ok in enumerate(elig):
        if ok:
            v = vals[i]
            if plateau and v != prev_val:
                flush()
            plateau.append(i)
            prev_val = v
        else:
            flush()
            prev_val = object()
    flush()
    return reps


class ExcursionPlanner:
    """Frozen stage-2 growth in ONE absolute pair-index space.

    Index-space note (frozen ruling): everything here runs in ONE space --
    absolute indices into ``measurement["frame_pairs"]``.  Plateau lists,
    blocked sets, grown runs and active ranges all share it; there is no
    per-run local numbering anywhere (review: mixed-index defect).

    Growth rules (stage 2, frozen):
      * a run starts at an eligible plateau (whole flat top) and grows
        outward over pairs that are valid, frame-contiguous and >= the
        ACTIVITY FLOOR (abs_threshold * activity_floor_ratio);
      * a valid pair strictly below the floor is RESTING: growth may cross
        a resting stretch only within the deterministic budget
        ``merge_gap_frames`` and only when EVERY skipped pair is valid,
        frame-contiguous and >= abs_threshold / 2 (the bridge floor --
        same spirit as E1b; low-but-valid interior may extend one peak's
        run but must never swallow a deep valley or a hole);
      * invalid measurements and confirmed-cut-crossing (blocked) pairs
        are HARD stops: nothing spans missing data or a cut;
      * non-contiguous real frame numbers stop growth (no faked spans).

    ``active_pair_range()`` returns the pair indices inside a grown run
    whose measurement is ACTIVE (>= the stage-1 absolute floor).  The R2
    interval mapping uses ONLY these anchors, so resting boundary pairs
    never fabricate event edges while still being covered by the grown
    run's interior.
    """

    def __init__(self, measurement: dict[str, Any], thr: float,
                 floor: float, merge_gap_frames: int,
                 blocked: set[int] | None = None):
        self.pairs = measurement["frame_pairs"]
        self.n = len(self.pairs)
        self.thr = float(thr)              # stage-1 absolute floor
        self.floor = float(floor)          # stage-2 activity floor
        self.bridge_min = self.thr / 2.0   # stage-3 bridge floor
        self.mgf = int(merge_gap_frames)
        self.blocked = set(blocked or ())
        self.vals = [_pair_value(measurement, i) for i in range(self.n)]

    # ---- predicates -----------------------------------------------------
    def _fn(self, i: int) -> int:
        return int(self.pairs[i]["frame_no"])

    def _contig(self, a: int, b: int) -> bool:
        """post frame of pair a is immediately followed by post of pair b."""
        return 0 <= a < self.n and 0 <= b < self.n \
            and self._fn(a) + 1 == self._fn(b)

    def _growable(self, j: int) -> bool:
        """valid, unblocked, contiguous with previous pair, >= activity floor"""
        return (0 <= j < self.n and self.vals[j] is not None
                and j not in self.blocked
                and self._contig(j - 1, j)
                and self.vals[j] >= self.floor)

    def _crossable_rest(self, lo: int, hi: int) -> bool:
        """May growth jump the resting stretch [lo..hi]?  Every skipped pair
        must be valid, unblocked, frame-contiguous and >= bridge floor;
        the stretch length must fit the merge_gap_frames budget."""
        if hi < lo or (hi - lo + 1) > self.mgf:
            return False
        for j in range(lo, hi + 1):
            if self.vals[j] is None or j in self.blocked \
                    or not self._contig(j - 1, j) \
                    or self.vals[j] < self.bridge_min:
                return False
        return True

    # ---- stage-2 growth --------------------------------------------------
    def extend(self, plateau: list[int]) -> tuple[int, int]:
        """Grow a run outward from a whole plateau.  Returns the inclusive
        grown pair range (absolute indices)."""
        lo, hi = plateau[0], plateau[-1]
        # left side
        while True:
            j = lo - 1
            if self._growable(j):
                lo = j
                continue
            if self.vals[j] is not None and j not in self.blocked \
                    and self._contig(j - 1, j) and self.vals[j] < self.floor:
                r = j
                while r - 1 >= 0 and self.vals[r - 1] is not None \
                        and self.vals[r - 1] < self.floor \
                        and (r - 1) not in self.blocked \
                        and self._contig(r - 2, r - 1):
                    r -= 1
                if self._growable(r - 1) \
                        and self._crossable_rest(r, lo - 1) \
                        and (lo - r) <= self.mgf:
                    lo = r - 1
                    continue
            break
        # right side
        while True:
            j = hi + 1
            if self._growable(j):
                hi = j
                continue
            if j < self.n and self.vals[j] is not None and j not in self.blocked \
                    and self._contig(j - 1, j) and self.vals[j] < self.floor:
                r = j
                while r + 1 < self.n and self.vals[r + 1] is not None \
                        and self.vals[r + 1] < self.floor \
                        and (r + 1) not in self.blocked \
                        and self._contig(r, r + 1):
                    r += 1
                if self._growable(r + 1) \
                        and self._crossable_rest(hi + 1, r) \
                        and (r - hi) <= self.mgf:
                    hi = r + 1
                    continue
            break
        return lo, hi

    def active_pair_range(self, start: int, end: int) -> list[int]:
        """Absolute pair indices inside grown run [start..end] carrying an
        ACTIVE measurement (>= stage-1 absolute floor)."""
        return [j for j in range(start, end + 1)
                if self.vals[j] is not None and self.vals[j] >= self.thr]


class _Segmentation(Exception):
    """Raised when a candidate cannot cover its interval contiguously
    (non-consecutive frame numbers): the run must segment, never fake a
    half-open span across a hole."""


def _contiguous_post_interval(pairs: list[dict[str, Any]], idxs: list[int],
                              ) -> tuple[int, int]:
    """R2 mapping: active pair indices -> half-open frame interval.

    The covered frame set is the POST frames of every ACTIVE pair: the
    interval starts at ``pairs[idxs[0]]["frame_no"]`` (the FIRST post
    frame; prev_frame_no of that pair is only the pre side of the change
    and is NOT part of this run's evidence) and ends right-open after the
    LAST post frame, ``pairs[idxs[-1]]["frame_no"] + 1``, so both boundary
    post frames stay inside [start, end).

    Coverage rule (frozen): every frame strictly between the first and
    last post frame must be covered by SOME valid pair in the grown run
    (its post frame lies within the run span).  A frame whose adjacent
    measurements are all invalid/absent (an undecodable hole) breaks
    coverage -- the caller segments instead of faking a span.
    """
    if not idxs:
        raise _Segmentation("empty active range")
    lo_i, hi_i = idxs[0], idxs[-1]
    start_fn = int(pairs[lo_i]["frame_no"])
    end_fn = int(pairs[hi_i]["frame_no"]) + 1
    # contiguity of the ACTIVE pairs themselves (no index holes inside
    # the anchor stretch) plus real frame-number continuity
    for x, y in zip(idxs, idxs[1:]):
        if int(pairs[y]["frame_no"]) != int(pairs[x]["frame_no"]) + 1:
            raise _Segmentation(
                f"active pair {y} post frame {pairs[y]['frame_no']} breaks "
                f"contiguity after {pairs[x]['frame_no']}")
    # every interior frame must be covered by a valid pair of the run
    covered = {int(pairs[j]["frame_no"])
               for j in range(lo_i, hi_i + 1) if pairs[j]["valid"]}
    for fn in range(start_fn, end_fn):
        if fn not in covered:
            raise _Segmentation(f"frame {fn} has no valid covering pair")
    return start_fn, end_fn


# ---------------------------------------------------------------------------
# stage 1: screening + prominence (pure, testable in isolation)
# ---------------------------------------------------------------------------

def _plateau_groups(pairs: list[dict[str, Any]], vals: list[float | None],
                    elig: list[bool]) -> list[list[int]]:
    """Group eligible indices into maximal plateaus: CONSECUTIVE indices
    (no ineligible index in between) sharing one identical value.

    Non-adjacent equal values -- e.g. the entry and exit spikes of a
    symmetric burst separated by resting pairs -- are NOT one flat top;
    "flat top" means spatially contiguous samples.  Ascending order.

    A plateau NEVER extends across a frame-number hole: pair i and pair
    i+1 belong to one flat top only when they are adjacent in the series
    AND their post frames are consecutive REAL frame numbers.  Two equal
    values measured around a missing/undecodable frame are distinct
    physical observations, not one flat top (frozen R2 contiguity).
    """
    groups: list[list[int]] = []
    cur: list[int] = []
    prev_ok = False
    for i, ok in enumerate(elig):
        if (ok and prev_ok and vals[i] == vals[cur[-1]]
                and int(pairs[i]["frame_no"])
                == int(pairs[cur[-1]]["frame_no"]) + 1):
            cur.append(i)
        else:
            if cur:
                groups.append(cur)
            cur = [i] if ok else []
        prev_ok = ok
    if cur:
        groups.append(cur)
    return groups


def find_motion_peaks(measurement: dict[str, Any], *,
                      params: dict[str, Any] | None = None,
                      blocked_pairs: set[int] | None = None,
                      ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (peaks, analysis).  Each peak dict carries pair_index,
    plateau_indices, value, pts_s, post/pre frame numbers, neighbor stats,
    single_sided flag.  Suppressed would-be peaks (blocked pairs above the
    absolute floor) are returned inside analysis['suppressed'] with their
    raw values kept -- never zeroed, never dropped."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    # frozen ruling 1: prominence_max_ratio is DEPRECATED / INACTIVE -- it
    # never participates in peak selection, so it is stripped from the
    # ACTIVE parameter set (provenance below echoes the active set only).
    p.pop("prominence_max_ratio", None)
    thr = float(p["abs_threshold"])
    prom = float(p["prominence_ratio"])
    win = int(p["window_frames"])

    pairs = measurement["frame_pairs"]
    n = len(pairs)
    vals = [_pair_value(measurement, i) for i in range(n)]
    blocked = set(blocked_pairs or ())

    suppressed: list[dict[str, Any]] = []
    for i in range(n):
        v = vals[i]
        if v is not None and i in blocked and v >= thr:
            suppressed.append({
                "pair_index": i,
                "prev_frame_no": pairs[i]["prev_frame_no"],
                "post_frame_no": pairs[i]["frame_no"],
                "pts_s": pairs[i]["pts_s"],
                "mean_abs_diff": _r(v),
                "max_abs_diff": pairs[i]["max_abs_diff"],
                "changed_pixel_ratio": pairs[i]["changed_pixel_ratio"],
            })

    # R1 (frozen ruling 4 -- WHOLE-PLATEAU GROUPING): candidates are formed
    # FIRST as maximal plateaus of CONSECUTIVE, equal-valued, VALID pairs;
    # only THEN is the whole plateau judged for eligibility as one unit.
    # Grouping by per-point over-threshold flags first is forbidden: it
    # would slice one flat top into several candidates.  A blocked pair
    # (confirmed-cut-crossing) breaks the flat top because it contributes
    # NO motion evidence to either side (R3); a pair that merely lacks PTS
    # keeps its valid measurement and NEVER splits a plateau (R4).
    # Neighbors exclude every member of the candidate's own plateau plus
    # all blocked pairs; both prominence clauses use the SINGLE frozen
    # formula (ruling 1).
    cand = [v is not None and i not in blocked
            for i, v in enumerate(vals)]
    prelim_groups = _plateau_groups(pairs, vals, cand)
    elig = [False] * n
    infos: dict[int, dict[str, Any]] = {}
    plateau_by_idx: dict[int, list[int]] = {}
    for sg in prelim_groups:
        gi = set(sg)
        pv = vals[sg[0]]
        if pv < thr:                       # absolute floor on the plateau
            continue
        lo = max(0, min(sg) - win)
        hi = min(n - 1, max(sg) + win)
        neigh = [vals[j] for j in range(lo, hi + 1)
                 if j not in gi and vals[j] is not None and j not in blocked]
        if not neigh:
            # no valid neighbor evidence at all -> not eligible (never a
            # silent pass); an EDGE plateau with one-sided evidence still
            # has neighbors on its open side and is handled below.
            continue
        mx = max(neigh)
        sm = sorted(neigh)
        med = sm[len(sm) // 2] if len(sm) % 2 else (sm[len(sm) // 2 - 1] + sm[len(sm) // 2]) / 2.0
        # single frozen prominence formula (frozen ruling 1, strict >):
        #     v > max(N) * prominence_ratio   AND   v > median(N)
        # The MAX clause carries the ratio; the MEDIAN clause is a raw
        # majority floor WITHOUT the ratio -- two different statistics,
        # two different roles.  Equal-strength neighbours defeat prominence
        # through the max clause (v > v*r is false for r < 1), which is
        # intended: a constant-translation platform stays peak-free apart
        # from its onset edge.
        if pv > mx * prom and pv > med:
            rep = sg[0] + (len(sg) - 1) // 2   # LEFT-MIDDLE rule (R1)
            for j in sg:
                elig[j] = True
                plateau_by_idx[j] = sg
                infos[j] = {"neighbor_max": _r(mx),
                            "neighbor_median": _r(med),
                            "window": {"start": lo, "end": hi},
                            "valid_samples": len(neigh),
                            # edge evidence: one side of the plateau has
                            # no observable pairs at all (series boundary)
                            "single_sided": (min(sg) - win < 0
                                             or max(sg) + win > n - 1)}

    reps = sorted({plateau_by_idx[j][0]
                   + (len(plateau_by_idx[j]) - 1) // 2
                   for j in range(n) if elig[j]})
    peaks: list[dict[str, Any]] = []
    for i in reps:
        plateau = plateau_by_idx[i]
        info = infos[i]
        peaks.append({
            "pair_index": i,
            "plateau_indices": plateau,
            "value": _r(float(vals[i])),
            "pts_s": None if pairs[i]["pts_s"] is None else _r(pairs[i]["pts_s"]),
            "post_frame_no": pairs[i]["frame_no"],
            "prev_frame_no": pairs[i]["prev_frame_no"],
            "neighbor_max": info["neighbor_max"],
            "neighbor_median": info["neighbor_median"],
            "window": info["window"],
            "valid_samples": info["valid_samples"],
            "single_sided_window": info["single_sided"],
        })

    analysis = {
        "threshold_applied": _r(thr),
        "prominence_ratio": _r(prom),
        "window_frames": win,
        "coverage_ratio": _r(float(measurement.get("coverage_ratio", 0.0))),
        "low_coverage": float(measurement.get("coverage_ratio", 0.0))
                        < float(p["min_coverage_ratio"]),
        "n_pairs": n,
        "n_valid_pairs": sum(1 for v in vals if v is not None),
        "suppressed": suppressed,
        "params": p,
    }
    return peaks, analysis


# ---------------------------------------------------------------------------
# stage 2: deterministic merging of peak-centered runs
# ---------------------------------------------------------------------------

def merge_motion_runs(runs: list[tuple[int, int]],
                      measurement: dict[str, Any],
                      threshold: float,
                      merge_gap_frames: int,
                      blocked_pairs: set[int] | None = None) -> list[list[int]]:
    """Group peak-centered activity runs into candidate groups.

    ``runs[k]`` is the activity run grown around ``peaks[k]["plateau"]``, so
    run order == peak order; grouping happens on that shared index space.

    Rules (deterministic, auditable, same bridge spirit as E1b):
      * overlapping/touching runs always merge;
      * separation of <= merge_gap_frames pairs merges only when EVERY
        intervening pair is valid and >= threshold/2;
      * otherwise the runs stay separate events (two independent bursts
        must not collapse through a resting plateau).
    Returns lists of run indices (ascending by run start).
    """
    if not runs:
        return []
    pairs = measurement["frame_pairs"]

    def val(j: int) -> float | None:
        pr = pairs[j]
        return None if not pr["valid"] else float(pr["mean_abs_diff"])

    # frozen merge-stage bridge floor: EVERY intervening pair must be valid
    # and >= abs_threshold/2 (same spirit as E1b).  This is deliberately
    # DISTINCT from the lower run-extension activity floor -- three stages,
    # three rules; a low-but-valid interior may extend a single peak's run
    # but must never glue two independent peaks across a deep valley.
    bridge_min = threshold / 2.0
    blocked = set(blocked_pairs or ())

    order = sorted(range(len(runs)), key=lambda k: (runs[k][0], runs[k][1]))
    groups: list[list[int]] = [[order[0]]]
    for k in order[1:]:
        # span of the whole current group, not just its last run
        hi_p = max(runs[m][1] for m in groups[-1])
        lo_c, _hi_c = runs[k]
        if lo_c <= hi_p + 1:
            groups[-1].append(k)
            continue
        gap = lo_c - hi_p - 1
        between = range(hi_p + 1, lo_c)
        bridged = (gap <= merge_gap_frames
                   and all(val(j) is not None and val(j) >= bridge_min
                           for j in between))
        if bridged:
            # hard stop regardless of budget: a confirmed-cut-crossing pair
            # or an unmeasurable pair may NEVER be glued over -- the events
            # segmentation below would split them anyway, so keep the runs
            # separate here too (deterministic, audit-consistent).
            if any(j in blocked for j in between):
                groups.append([k])
            else:
                groups[-1].append(k)
        else:
            groups.append([k])
    return groups


# ---------------------------------------------------------------------------
# stage 3: document assembly
# ---------------------------------------------------------------------------

def extract_peak_motion_events(
    measurement: dict[str, Any],
    shots: list[ShotSpan],
    *,
    video_id: str,
    source_sha256: str,
    params: dict[str, Any] | None = None,
    sha_by_fn: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Apply the frozen E2b criteria to an E2a motion record and emit an
    independent ``openmontage.shot_events.v1`` document (CANDIDATE only)."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    # frozen ruling 1: the deprecated prominence_max_ratio never joins the
    # ACTIVE parameter set echoed into provenance / measurement_provenance.
    p.pop("prominence_max_ratio", None)

    pairs = measurement["frame_pairs"]
    fn_to_idx = {int(pr["frame_no"]): i for i, pr in enumerate(pairs)}
    confirmed_edges = _confirmed_cut_frame_nos(shots)
    # Frozen adjacency definition (identical to E2a's marker semantics):
    # a confirmed cut at frame number b blocks EXACTLY the pair whose real
    # frame numbers are (b-1, b).  The mapping is done through prev_frame_no
    # / frame_no values only -- never through list positions -- so removed
    # or non-contiguous frames cannot shift the verdict (review check 1).
    blocked: set[int] = set()
    for i, pr in enumerate(pairs):
        if int(pr["frame_no"]) in confirmed_edges \
                and int(pr["prev_frame_no"]) == int(pr["frame_no"]) - 1:
            blocked.add(i)

    peaks, analysis = find_motion_peaks(measurement, params=p,
                                        blocked_pairs=blocked)

    # structural audit (R3): every confirmed-cut-crossing pair whose raw
    # value is at/above the absolute floor must be reported in
    # audit.suppressed_peaks even when it never wins local prominence --
    # suppression is a boundary verdict, not a numeric accident.
    thr_abs = float(p["abs_threshold"])
    hard_bds = [bd for s in shots for bd in (s.start_boundary, s.end_boundary)
                if bd is not None and bd.is_hard_cut()]

    def _owning_boundary(pr: dict[str, Any]):
        """The confirmed edge this pair straddles: post frame == b (the
        shot START edge) or pre frame == b (the previous shot's END edge)."""
        for bd in hard_bds:
            if int(bd.frame_no) in (int(pr["frame_no"]), int(pr["prev_frame_no"])):
                return bd
        return None

    known_sup = {sp["pair_index"] for sp in analysis["suppressed"]}
    for sp in analysis["suppressed"]:
        bd = _owning_boundary(pairs[sp["pair_index"]])
        sp["reason"] = "adjacent_to_confirmed_hard_cut"
        sp["boundary_support"] = _jsonable(asdict(bd)) if bd is not None else None
    for j in sorted(blocked):
        if j in known_sup:
            continue
        v = _pair_value(measurement, j)
        if v is not None and v >= thr_abs:
            pr = pairs[j]
            entry = {
                "pair_index": j,
                "prev_frame_no": pr["prev_frame_no"],
                "post_frame_no": pr["frame_no"],
                "pts_s": pr["pts_s"],
                "mean_abs_diff": _r(v),
                "max_abs_diff": pr["max_abs_diff"],
                "changed_pixel_ratio": pr["changed_pixel_ratio"],
                "reason": "adjacent_to_confirmed_hard_cut",
            }
            bd = _owning_boundary(pr)
            entry["boundary_support"] = (_jsonable(asdict(bd))
                                         if bd is not None else None)
            analysis["suppressed"].append(entry)

    floor = float(p["abs_threshold"]) * float(p["activity_floor_ratio"])
    mgf = int(p["merge_gap_frames"])
    # one activity run per peak plateau, grown by the ExcursionPlanner in
    # ONE absolute pair-index space (stage-2 frozen rules: activity floor
    # abs_threshold * activity_floor_ratio; resting stretches crossable
    # only within the merge_gap_frames budget AND only when every skipped
    # pair is valid and >= abs_threshold/2; invalid measurements and
    # confirmed-cut-crossing pairs are hard stops).
    planner = ExcursionPlanner(measurement, thr_abs, floor, mgf,
                               blocked=blocked)

    def _shot_of(fn: int) -> ShotSpan | None:
        for s in shots:
            if s.owns_frame(fn):
                return s
        return None

    def _segment_runs(runs_in: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Split every grown run at hard-stop pairs (confirmed-cut-crossing
        or unmeasurable/invalid pairs).  A stop pair belongs to NO segment;
        segments never silently span a hole or a cut, and each side keeps
        its own evidence (frozen segmentation ruling)."""
        out: list[tuple[int, int]] = []
        for lo, hi in runs_in:
            cur: int | None = None
            for j in range(lo, hi + 1):
                stop = (j in blocked) or (_pair_value(measurement, j) is None)
                if stop:
                    if cur is not None:
                        out.append((cur, j - 1))
                    cur = None
                elif cur is None:
                    cur = j
            if cur is not None:
                out.append((cur, hi))
        return out

    # ---- stage 2: grow one run per peak, then split at hard stops -------
    # Excursion anchoring (frozen): a grown run is ANCHORED at its own
    # plateau; it may absorb an equal-valued symmetric spike beyond a
    # resting interior (entry+exit of ONE burst = one candidate) but a
    # FOREIGN plateau (different value) always stays a separate anchor --
    # two independent bursts must not collapse through their shared rest.
    # Runs live in ONE absolute pair-index space; every downstream lookup
    # resolves a peak through ITS OWN run list (never peaks.index()
    # reverse lookups or mixed index spaces -- review: mixed-index defect).
    flat: list[tuple[int, int]] = []          # unique grown/merged runs
    run_peaks: dict[tuple[int, int], list[int]] = {}   # run -> peak indices
    peak_run_seg: dict[int, list[tuple[int, int]]] = {}  # peak -> segments
    for pi, pk in enumerate(peaks):
        lo, hi = planner.extend(pk["plateau_indices"])
        # absorb already-planned runs whose only active pairs form an
        # equal-valued plateau of THIS peak and fit inside the grown span
        for (rlo, rhi), rpks in list(run_peaks.items()):
            if rlo >= lo and rhi <= hi and rpks:
                other = peaks[rpks[0]]
                if other["value"] == pk["value"]:
                    lo, hi = min(lo, rlo), max(hi, rhi)
        key = (lo, hi)
        absorbed = [k for k, v in run_peaks.items()
                    if k != key and lo <= k[0] and k[1] <= hi]
        merged_pks = sorted({pi} | {q for k in absorbed for q in run_peaks[k]}
                            | {q for k, v in run_peaks.items()
                               if k == key for q in v})
        for k in absorbed:
            del run_peaks[k]
            flat.remove(k)
        run_peaks[key] = merged_pks
        if key not in flat:
            flat.append(key)
    flat_sorted = sorted(flat)
    # recompute per-peak segment lists AFTER merging (hard stops split each
    # final run; every peak of a run maps through that run's own segments)
    seg_of_run: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for key in flat_sorted:
        seg_of_run[key] = _segment_runs([key])
    for pi in range(len(peaks)):
        segs: list[tuple[int, int]] = []
        for key, rpks in run_peaks.items():
            if pi in rpks:
                segs.extend(seg_of_run[key])
        peak_run_seg[pi] = sorted(segs)
    groups = merge_motion_runs(flat_sorted, measurement, thr_abs, mgf,
                               blocked_pairs=blocked)

    events: list[dict[str, Any]] = []
    audit_no_pts: list[dict[str, Any]] = []
    extra_gaps: list[dict[str, Any]] = []

    def _no_pts_entry(sh_id: str | None, pk: dict[str, Any],
                      start_fn: int | None, end_fn: int | None,
                      detail: str) -> dict[str, Any]:
        gap_start = (pk["post_frame_no"] if start_fn is None else start_fn)
        gap_end = (pk["post_frame_no"] + 1 if end_fn is None else end_fn)
        return {
            "shot_id": sh_id,
            "peak_post_frame_no": pk["post_frame_no"],
            "peak_pair_index": pk["pair_index"],
            "peak_value": pk["value"],
            "interval_frames": {"domain": "frame_no",
                                 "start": start_fn if start_fn is not None
                                 else pk["post_frame_no"],
                                 "end": end_fn if end_fn is not None
                                 else pk["post_frame_no"] + 1},
            "gaps": [{"reason": "missing_pts",
                      "frame_range": {"domain": "frame_no",
                                      "start": gap_start, "end": gap_end},
                      "detail": detail}],
        }

    def _emit(sh: ShotSpan, seg_peaks: list[dict[str, Any]],
              lo: int, hi: int, start_fn: int, end_fn: int,
              merged_runs: list[list[int]]) -> None:
        """Assemble one CANDIDATE event for a segmented, attributed run."""
        # ---- R4 (frozen ruling 3): the STRONGEST peak is chosen by value
        # FIRST; ties are broken deterministically toward the EARLIEST pair
        # index (entry-peak anchor).  Only AFTER the peak is selected is its
        # PTS checked: if that peak lacks an authoritative PTS the candidate
        # is withheld -- never silently re-anchored to a weaker peak.  A
        # weaker peak missing PTS does NOT veto the event, but must stay
        # visible in the peak evidence (its pts_s=None) and is recorded in
        # the audit trail.
        peak = min(seg_peaks, key=lambda pk: (-pk["value"], pk["pair_index"]))
        weaker_no_pts = [pk for pk in seg_peaks
                         if pk is not peak and pk["pts_s"] is None]

        # ---- R4: strongest peak without authoritative PTS -> no event
        if peak["pts_s"] is None:
            entry = _no_pts_entry(
                sh.shot_id, peak, start_fn, end_fn,
                "strongest peak lacks authoritative PTS; candidate withheld "
                "(no fabricated time)")
            if weaker_no_pts:
                entry["weaker_peaks_without_pts"] = [
                    {"pair_index": wk["pair_index"],
                     "post_frame_no": wk["post_frame_no"],
                     "value": wk["value"]} for wk in weaker_no_pts]
            audit_no_pts.append(entry)
            return

        # interval_pts from real measured PTS values only
        i_lo = next(i for i in range(len(pairs))
                    if pairs[i]["frame_no"] >= start_fn)
        i_hi = next(i for i in range(len(pairs) - 1, -1, -1)
                    if pairs[i]["frame_no"] < end_fn)
        pts_vals = [pairs[i]["pts_s"] for i in range(i_lo, i_hi + 1)
                    if pairs[i]["pts_s"] is not None]
        if not pts_vals:
            audit_no_pts.append(_no_pts_entry(
                sh.shot_id, peak, start_fn, end_fn,
                "no usable PTS inside candidate interval"))
            return
        iv_pts_start, iv_pts_end = _r(min(pts_vals)), _r(max(pts_vals))

        # ---- boundary context & review reasons (R3: no fake enums)
        boundary_ctx: dict[str, Any] = {}
        review_reasons: list[str] = []
        if peak["single_sided_window"]:
            review_reasons.append("single_sided_window")
        near_unconfirmed = False
        for side, bd in (("start_boundary", sh.start_boundary),
                         ("end_boundary", sh.end_boundary)):
            if bd is None:
                continue
            boundary_ctx[side] = _jsonable(asdict(bd))
            if not bd.is_hard_cut():
                for pk in seg_peaks:
                    if pk["pair_index"] in (fn_to_idx.get(bd.frame_no),
                                            fn_to_idx.get(bd.frame_no, -1) - 1):
                        near_unconfirmed = True
        if near_unconfirmed:
            review_reasons.append("unconfirmed_boundary_neighbor")

        # ---- brightness-transition heuristic (retained + flagged).
        # Frozen ruling: high changed-pixel ratio AND max_abs_diff close to
        # mean_abs_diff (whole-frame uniform change, no spatial structure)
        # -> descriptive marker only, never a semantic classification.
        thr_b = float(p["abs_threshold"])
        ratio_min = float(p["brightness_like_ratio_min"])
        max_over_mean = float(p["brightness_like_max_over_mean"])

        def _looks_uniform_frame_change(pk: dict[str, Any]) -> bool:
            prq = pairs[pk["pair_index"]]
            cpr, mxd = prq["changed_pixel_ratio"], prq["max_abs_diff"]
            if pk["value"] < thr_b or cpr is None or mxd is None:
                return False
            if cpr < ratio_min:
                return False
            return mxd <= max(pk["value"], 1e-9) * max_over_mean

        bright_like = any(_looks_uniform_frame_change(pk) for pk in seg_peaks)
        if bright_like:
            review_reasons.append("possible_brightness_transition")

        base_meas: dict[str, Any] = {
            "metric": "mean_abs_diff",
            "active_pair_range": [lo, hi],
            "merged_run_ranges": merged_runs,
            "peaks": [{k: pk[k] for k in
                       ("pair_index", "post_frame_no", "value", "pts_s",
                        "plateau_indices", "neighbor_max",
                        "neighbor_median", "single_sided_window")}
                      for pk in seg_peaks],
            "threshold_applied": analysis["threshold_applied"],
            "prominence_ratio": analysis["prominence_ratio"],
            "window_frames": analysis["window_frames"],
            "coverage_ratio": analysis["coverage_ratio"],
            "duration_frames": end_fn - start_fn,
            "duration_s": _r(iv_pts_end - iv_pts_start),
            "valid_samples_in_window": peak["valid_samples"],
            "possible_brightness_transition": bright_like,
            "review_reasons": sorted(set(review_reasons)),
        }

        ref_idxs: list[int] = []
        for pk in seg_peaks:
            ref_idxs += [pk["pair_index"], pk["pair_index"] + 1]
        refs, ref_gaps = _frame_refs(
            {"frame_nos": [int(pr["frame_no"]) for pr in pairs],
             "pts_s": [pr["pts_s"] for pr in pairs]},
            sorted(set(ref_idxs)), sha_by_fn)
        ev_gaps = list(measurement["gaps"]) + ref_gaps

        ev = Event(
            event_id="EVT_0000",
            shot_id=sh.shot_id,
            event_type="peak_motion",
            status="CANDIDATE",
            interval_frames=Interval("frame_no", start_fn, end_fn),
            interval_pts=Interval("pts_s", iv_pts_start, iv_pts_end),
            peak_pts_s=peak["pts_s"],
            confidence=None,
            confidence_note=("calibration benchmark pending (Commit F); "
                              "trust status+evidence, not a scalar"),
            evidence=EventEvidence(
                source="l1_5_measurement",
                frame_refs=refs,
                measurements=base_meas,
                baseline={
                    "neighbor_max": peak["neighbor_max"],
                    "neighbor_median": peak["neighbor_median"],
                    "window": peak["window"],
                },
                gaps=ev_gaps,
                boundary_context=boundary_ctx,
            ),
        )
        validate_event(ev, shot=sh)
        events.append(ev.to_dict())

    for group in groups:
        g_runs = [flat_sorted[k] for k in group]
        gs_lo = min(a for a, _ in g_runs)
        gs_hi = max(b for _, b in g_runs)
        # peaks whose grown run participates in this group -- resolved via
        # the explicit peak -> segment maps built in the SAME absolute pair
        # index space (never a reverse peaks.index() lookup against merged
        # runs; review: mixed-index defect).  A peak joins when ANY of its
        # post-hard-stop segments overlaps the group span.
        g_peaks = [pk for pi, pk in enumerate(peaks)
                   if any(not (b < gs_lo or a > gs_hi)
                          for a, b in peak_run_seg[pi])]
        if not g_peaks:
            continue

        # ---- segmentation: any blocked (confirmed-cut-crossing) pair OR
        # any unmeasurable (invalid) pair between two runs forces a split;
        # events may never span a confirmed cut and never silently span
        # missing data.  Collect every hard-stop pair inside the group's
        # total span; each one forces a split of the candidate interval --
        # never a silent span across it.
        stops = sorted(j for j in range(gs_lo, gs_hi + 1)
                       if j in blocked or _pair_value(measurement, j) is None)
        seg_bounds: list[tuple[int, int]] = []  # (lo_pair, hi_pair) inclusive
        cur_lo = gs_lo
        for s in stops:
            if s > cur_lo:
                seg_bounds.append((cur_lo, s - 1))
            cur_lo = s + 1
        if cur_lo <= gs_hi:
            seg_bounds.append((cur_lo, gs_hi))

        for lo, hi in seg_bounds:
            # R2 anchors: ACTIVE pairs (>= stage-1 absolute floor) inside
            # this segment.  Resting interior pairs are covered by the
            # interval but never fabricate an event edge on their own.
            anchors = planner.active_pair_range(lo, hi)
            anchors = [j for j in anchors if j not in blocked]
            if not anchors:
                continue
            try:
                start_fn, end_fn = _contiguous_post_interval(pairs, anchors)
            except _Segmentation:
                extra_gaps.append({
                    "reason": "decode_misalignment",
                    "frame_range": {"domain": "frame_no",
                                    "start": pairs[anchors[0]]["frame_no"],
                                    "end": pairs[anchors[-1]]["frame_no"] + 1},
                    "detail": "active pairs do not cover a contiguous frame-number "
                              "range; candidate segmented instead of faked",
                })
                continue

            # E1b post-frame attribution rule: EVERY covered post frame must
            # belong to the same shot -- an activity range that would touch
            # two shots can only be explained by a boundary crossing, so we
            # attribute by its FIRST post frame and clip at the shot edge.
            sh = _shot_of(start_fn)
            if sh is None or start_fn < sh.start_frame_no or end_fn > sh.end_frame_no:
                extra_gaps.append({
                    "reason": "boundary_ambiguity",
                    "frame_range": {"domain": "frame_no",
                                    "start": start_fn, "end": end_fn},
                    "detail": "candidate frames fall outside every single shot "
                              "span; cannot attribute intra-shot",
                })
                continue
            # clip to shot ownership (half-open), then re-verify contiguity
            start_fn = max(start_fn, sh.start_frame_no)
            end_fn = min(end_fn, sh.end_frame_no)
            if start_fn >= end_fn:
                continue

            seg_peaks = [pk for pk in g_peaks
                         if start_fn <= pk["post_frame_no"] < end_fn]
            if not seg_peaks:
                continue
            # per-peak sub-runs actually covered by THIS segment -- the
            # frozen E2b merge evidence: several peak-centered runs joined
            # into one candidate interval without losing their identity.
            merged_runs = []
            for pk in seg_peaks:
                for rlo, rhi in peak_run_seg[peaks.index(pk)]:
                    mlo, mhi = max(rlo, lo), min(rhi, hi)
                    if mlo <= mhi:
                        merged_runs.append([mlo, mhi])
            merged_runs.sort()
            _emit(sh, seg_peaks, lo, hi, start_fn, end_fn, merged_runs)

    # deterministic event-id order: by interval start, then end, then shot
    events.sort(key=lambda e: (e["interval_frames"]["start"],
                               e["interval_frames"]["end"], e["shot_id"]))
    for i, e in enumerate(events, 1):
        e["event_id"] = f"EVT_{i:04d}"

    prov = AlgorithmProvenance(
        name=ALGORITHM_NAME, algorithm_version=ALGORITHM_VERSION,
        params=p,
        input_artifacts={"measurement_input_digest": measurement["input_digest"]})
    quality_reason = None
    if analysis["low_coverage"]:
        quality_reason = (f"measurement coverage_ratio {analysis['coverage_ratio']} "
                          f"< min {p['min_coverage_ratio']}; candidates withheld "
                          f"from PASS")
    doc = build_events_document(
        video_id=video_id, source_sha256=source_sha256,
        shots=[_shot_span_dict(s) for s in shots], events=events,
        algorithm=prov)
    doc["quality_gate"] = {
        "status": "REVIEW",
        "reason": quality_reason or (
            "no calibration benchmark yet; PASS requires explicit quality "
            "conditions (Commit F)")}
    doc["measurement_provenance"] = {
        "algorithm": measurement["provenance"]["name"],
        "algorithm_version": measurement["algorithm_version"],
        "input_digest": measurement["input_digest"],
        # extractor policy actually applied to this document (same style as
        # the E1b brightness document; thresholds are declared policy, not
        # cross-material truth)
        "algorithm_params": p,
    }
    doc["audit"] = {
        # reason + owning BoundarySupport were attached structurally above;
        # nothing here depends on prominence outcomes (R3).
        "suppressed_peaks": list(analysis["suppressed"]),
        "candidates_without_peak_pts": audit_no_pts,
        "unattributed_ranges": extra_gaps,
        "measurement_gaps": measurement["gaps"],
    }
    return doc
