"""L1.5 Commit E (E2b): peak_motion candidate extraction over E2a measures.

Layering honored: E2a (``event_motion.compute_motion_sequence``) produced the
raw frame-pair picture-change sequence; THIS module only applies explicit,
auditable criteria to that record.  It never re-decodes video, never
recomputes pixel differences, performs no disk writes and never touches L2
events.json.

v1.0.0 criteria contract (frozen in review, revisions R1-R4 applied):

  Peak eligibility (R1):
    * absolute floor: mean_abs_diff >= abs_threshold (params, provenance);
    * local prominence: value > max(neighbors) * prominence_ratio AND
      value > median(neighbors), where neighbors are the valid, UNSUPPRESSED
      pairs within +/- window_frames -- the plateau itself is EXCLUDED;
    * flat top: consecutive equal-valued qualifying pairs form ONE plateau
      whose representative index is start + (len-1)//2 (LEFT-MIDDLE for even
      lengths) -- deterministic regardless of traversal order;
    * edge peaks are allowed but carry ``single_sided_window`` evidence;
    * a pair adjacent to a CONFIRMED hard cut (BoundarySupport.is_hard_cut()
      -- the ONLY source of confirmed semantics, no new enums, R3) is
      peak-INELIGIBLE: it is recorded verbatim in
      ``audit.suppressed_peaks`` with its boundary evidence attached.

  Activity runs and merging:
    * a run starts at each eligible peak and extends outward while adjacent
      pairs are valid and >= activity_floor_ratio * abs_threshold;
    * extension stops at invalid pairs (R: nothing spans missing data), at
      non-contiguous frame numbers (real frame_no gaps), and at confirmed
      cut-crossing pairs (forced segmentation);
    * two peaks merge into one candidate only when their runs overlap or are
      separated by <= merge_gap_frames intervening pairs, EVERY one of which
      is valid and >= abs_threshold/2 (bridge rule, same spirit as E1b).

  Candidate interval (R2, post-frame semantics):
    * active pair range [a..b] covers post frames a+1 .. b+1; the half-open
      event interval is [frame_no(a+1), frame_no(b+1) + 1) for contiguous
      frame numbers; non-contiguous coverage NEVER collapses into a fake
      +1 span (the run segments instead);
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
#: silently baked into results.
DEFAULT_PARAMS: dict[str, Any] = {
    # absolute floor on mean_abs_diff (uint8 units) for peak eligibility
    "abs_threshold": 5.0,
    # dual-condition prominence (frozen contract, strict >): the plateau
    # value must exceed BOTH mean(neighbors)*prominence_ratio AND
    # max(neighbors)*prominence_max_ratio -- the second term keeps a spike
    # adjacent to an equally strong spike (burst entry/exit pair) eligible
    "prominence_ratio": 0.5,
    "prominence_max_ratio": 0.9,
    # fixed neighborhood window (pairs) used for prominence stats
    "window_frames": 3,
    # run extension continues while value >= activity_floor_ratio*abs_thr
    "activity_floor_ratio": 0.1,
    # max gap between peaks merged into one candidate (bridged runs)
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


def _bridgeable_rest(m: dict[str, Any], lo_lim: int, hi_lim: int,
                     blocked: set[int], floor: float,
                     merge_gap_frames: int, *, side: str, edge: int,
                     ) -> int | None:
    """Deterministic bounded bridge over a resting stretch (stage 2).

    From the run edge ``edge`` looking outward on ``side`` ("left"/"right"),
    count consecutive VALID but sub-floor pairs.  The stretch is bridgeable
    only when ALL hold:
      * length is between 1 and ``merge_gap_frames`` inclusive;
      * every pair in the stretch is frame-number contiguous with the run;
      * immediately beyond the stretch there is an ACTIVE pair (valid and
        >= floor) -- bridging into silence is meaningless and forbidden;
      * no invalid pair and no confirmed-cut-crossing pair sits inside the
        stretch (hard stops: those segment, they are never "rest").
    Returns the index of the active landing pair beyond the stretch, or
    None when no bridge applies.  Invalid/blocked pairs NEVER consume the
    rest budget -- they abort the bridge outright.
    """
    pairs = m["frame_pairs"]
    step = -1 if side == "left" else 1

    def contig(a: int, b: int) -> bool:
        """post frame of pair a is immediately followed by post of pair b."""
        return 0 <= a < len(pairs) and 0 <= b < len(pairs) \
            and int(pairs[a]["frame_no"]) + 1 == int(pairs[b]["frame_no"])

    j = edge + step
    cnt = 0
    while lo_lim <= j <= hi_lim:
        v = _pair_value(m, j)
        if j in blocked or v is None:
            return None                      # hard stop, not rest
        if not contig(j - step, j):
            return None                      # frame-number hole: never bridge
        if v >= floor:
            if cnt == 0 or cnt > merge_gap_frames:
                return None                  # no rest crossed, or too much
            return j                         # active landing pair
        cnt += 1
        if cnt > merge_gap_frames:
            return None
        j += step
    return None                              # ran off the series


def _extend_run(m: dict[str, Any], lo_lim: int, hi_lim: int,
                floor: float, blocked: set[int], plateau: list[int],
                merge_gap_frames: int,
                ) -> tuple[int, int]:
    """Grow the activity run around the peak's whole plateau (stage 2).

    Frozen E2b semantics -- THREE DISTINCT STAGES, three distinct rules:
      stage 1 peak eligibility   : abs_threshold (+ prominence);
      stage 2 run extension      : continue while adjacent pairs are valid,
        frame-contiguous and >= ACTIVITY FLOOR (abs_threshold *
        activity_floor_ratio); a valid sub-floor pair is RESTING and stops
        growth EXCEPT via the deterministic bounded bridge below;
      stage 3 run merging        : deeper floor (abs_threshold / 2), see
        merge_motion_runs -- a low-but-valid interior may extend ONE peak's
        run but must never glue TWO independent peaks across a deep valley.

    Bounded bridge (stage 2): at most ``merge_gap_frames`` consecutive
    valid resting pairs may be crossed, and only when an active, frame-
    contiguous pair lies immediately beyond the stretch -- so the entry
    spike, resting interior and exit spike of ONE burst form one excursion,
    while two bursts separated by a wider rest plateau stay separate.
    Hard stops: invalid measurements, confirmed-cut-crossing pairs and
    frame-number holes (never counted as rest).

    The plateau itself is always part of the run; blocking only stops
    EXTENSION -- a suppressed cross-cut pair must never silently delete
    measured evidence from the audit trail.
    """
    lo, hi = plateau[0], plateau[-1]

    def active(j: int) -> bool:
        v = _pair_value(m, j)
        return v is not None and v >= floor

    # leftward growth
    while True:
        if lo - 1 < lo_lim or (lo - 1) in blocked \
                or _pair_value(m, lo - 1) is None:
            break
        if active(lo - 1):
            lo -= 1
            continue
        land = _bridgeable_rest(m, lo_lim, hi_lim, blocked, floor,
                                merge_gap_frames, side="left", edge=lo)
        if land is None:
            break
        lo = land
    # rightward growth (symmetric)
    while True:
        if hi + 1 > hi_lim or (hi + 1) in blocked \
                or _pair_value(m, hi + 1) is None:
            break
        if active(hi + 1):
            hi += 1
            continue
        land = _bridgeable_rest(m, lo_lim, hi_lim, blocked, floor,
                                merge_gap_frames, side="right", edge=hi)
        if land is None:
            break
        hi = land
    return lo, hi


class _Segmentation(Exception):
    """Raised when a candidate cannot cover its interval contiguously
    (non-consecutive frame numbers): the run must segment, never fake a
    half-open span across a hole."""


def _contiguous_post_interval(pairs: list[dict[str, Any]], a: int, b: int,
                              ) -> tuple[int, int]:
    """R2 mapping: active pair range [a..b] -> post frames a+1..b+1.

    Pair i connects (prev_i, cur_i); the post frames covered by the whole
    range are prev_{a} (= frame a) through cur_{b} (= frame b+1), which is
    exactly pairs[a]["prev_frame_no"] .. pairs[b]["frame_no"].  The half-open
    interval therefore ends at pairs[b]["frame_no"] + 1 so that the LAST
    post frame stays INSIDE [start, end).  Contiguity of the underlying
    frame numbers is verified; a hole forces segmentation, never a faked
    span.
    """
    # The covered frame set is the POST frames of every pair in [a..b]:
    # pairs[a].frame_no .. pairs[b].frame_no.  prev_frame_no is only the
    # left anchor and is NOT part of the observed change evidence -- the
    # pre frame is static with respect to this run (frozen R2 semantics).
    start_fn = int(pairs[a]["frame_no"])
    end_fn = int(pairs[b]["frame_no"]) + 1
    cur = start_fn
    for j in range(a, b + 1):
        if int(pairs[j]["frame_no"]) != cur:
            raise _Segmentation(
                f"pair {j} post frame {pairs[j]['frame_no']} breaks "
                f"contiguity at {cur}")
        cur += 1
    return start_fn, end_fn


# ---------------------------------------------------------------------------
# stage 1: screening + prominence (pure, testable in isolation)
# ---------------------------------------------------------------------------

def _plateau_groups(vals: list[float | None], elig: list[bool],
                    ) -> list[list[int]]:
    """Group eligible indices into maximal plateaus: CONSECUTIVE indices
    (no ineligible index in between) sharing one identical value.

    Non-adjacent equal values -- e.g. the entry and exit spikes of a
    symmetric burst separated by resting pairs -- are NOT one flat top;
    "flat top" means spatially contiguous samples.  Ascending order.
    """
    groups: list[list[int]] = []
    cur: list[int] = []
    prev_ok = False
    for i, ok in enumerate(elig):
        if ok and prev_ok and vals[i] == vals[cur[-1]]:
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
    thr = float(p["abs_threshold"])
    prom = float(p["prominence_ratio"])
    prom_max = float(p["prominence_max_ratio"])
    win = int(p["window_frames"])

    pairs = measurement["frame_pairs"]
    n = len(pairs)
    vals = [_pair_value(measurement, i) for i in range(n)]
    blocked = set(blocked_pairs or ())

    suppressed: list[dict[str, Any]] = []
    over_thr: list[bool] = []
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
        over_thr.append(v is not None and i not in blocked and v >= thr)

    # R1 (review condition 1): the PLATEAU as a whole participates in peak
    # eligibility -- the representative position never changes the verdict.
    # Neighbors exclude every member of the candidate plateau (not just the
    # representative) plus all blocked pairs; both prominence conditions are
    # evaluated on that plateau-wide neighborhood.
    prelim_groups = _plateau_groups(vals, over_thr)
    elig = [False] * n
    infos: dict[int, dict[str, Any]] = {}
    for g in prelim_groups:
        gi, gj = set(g), None
        pv = vals[g[0]]
        lo = max(0, min(g) - win)
        hi = min(n - 1, max(g) + win)
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
        # dual condition: plateau value must exceed BOTH the window mean
        # scaled by prominence_ratio AND the strongest neighbor scaled by
        # prominence_max_ratio (strict >, frozen contract).
        if pv > med * prom and pv > mx * prom_max:
            for j in g:
                elig[j] = True
                infos[j] = {"neighbor_max": _r(mx), "neighbor_median": _r(med),
                            "window": {"start": lo, "end": hi},
                            "valid_samples": len(neigh),
                            # edge evidence: one side of the plateau has
                            # no observable pairs at all (series boundary)
                            "single_sided": (min(g) - win < 0
                                             or max(g) + win > n - 1)}

    reps = [g[0] + (len(g) - 1) // 2 for g in _plateau_groups(vals, elig)]
    plateau_by_rep: dict[int, list[int]] = {
        g[0] + (len(g) - 1) // 2: g for g in _plateau_groups(vals, elig)}
    peaks: list[dict[str, Any]] = []
    for i in reps:
        plateau = plateau_by_rep[i]
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

def merge_motion_runs(peaks: list[dict[str, Any]],
                      runs: list[tuple[int, int]],
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
    n = len(pairs)
    # one activity run per peak plateau, extended symmetrically outward.
    # Run growth is bounded by the ACTIVITY FLOOR (abs_threshold *
    # activity_floor_ratio): a valid pair strictly below the floor is a
    # resting frame pair and stops growth -- otherwise every burst would
    # bridge through zeros and swallow the whole shot.  Low-but-valid
    # pairs at/above the floor may be crossed within the deterministic
    # gap budget (see _extend_run).  Invalid measurements and confirmed-
    # cut-crossing pairs are hard stops.
    # NOTE: _shot_of / _clip_to_shot are nested closures created strictly
    # before any use (review of the earlier UnboundLocalError incident).
    def _shot_of(fn: int) -> ShotSpan | None:
        for s in shots:
            if s.owns_frame(fn):
                return s
        return None

    def _clip_to_shot(lo: int, hi: int) -> tuple[int, int]:
        """Never let a grown run cross a confirmed hard cut: the crossing
        pair is excluded and the side keeps its own evidence (post-frame
        attribution)."""
        while lo in blocked and lo < hi:
            lo += 1
        while hi in blocked and hi > lo:
            hi -= 1
        return lo, hi

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

    runs: list[tuple[int, int]] = []
    for pk in peaks:
        lo, hi = _extend_run(measurement, 0, n - 1, floor, blocked,
                             pk["plateau_indices"], mgf)
        # clip against shot ownership bounds too: a run may not grow past
        # the series edges owned by any single shot context; the explicit
        # per-shot segmentation below still enforces ShotSpan ownership.
        runs.extend(_segment_runs([_clip_to_shot(lo, hi)]))

    groups = merge_motion_runs(peaks, runs, measurement,
                               float(p["abs_threshold"]), mgf,
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
              lo: int, hi: int, start_fn: int, end_fn: int) -> None:
        """Assemble one CANDIDATE event for a segmented, attributed run."""
        # ---- R4 (frozen ruling a): the STRONGEST peak is chosen by value
        # with the deterministic tie rule FIRST; if that peak lacks an
        # authoritative PTS the candidate is withheld -- never silently
        # re-anchored to a weaker peak.  A weaker peak missing PTS does NOT
        # veto the event, but must stay visible in the peak evidence (its
        # pts_s=None) and is recorded in the audit trail.
        peak = max(seg_peaks, key=lambda pk: (pk["value"], pk["pair_index"]))
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
        g_runs = [runs[k] for k in group]
        # peaks whose plateau lies inside any run of the group (run index
        # space == peak index space, so `k` is directly a peak index)
        g_peaks = [pk for k in group for pk in [peaks[k]]
                   if any(lo <= j <= hi for lo, hi in g_runs
                          for j in pk["plateau_indices"])]
        if not g_peaks:
            continue

        # ---- segmentation: any blocked (confirmed-cut-crossing) pair OR
        # any unmeasurable (invalid) pair between two runs forces a split;
        # events may never span a confirmed cut and never silently span
        # missing data.  A blocked/invalid pair INSIDE one run cannot
        # happen by construction (_extend_run refuses to grow through it).
        # Collect every hard-stop pair (confirmed-cut crossing or
        # unmeasurable) inside the group's total span; each one forces a
        # split of the candidate interval -- never a silent span across it.
        g_lo = min(lo for lo, _ in g_runs)
        g_hi = max(hi for _, hi in g_runs)
        stops = sorted(j for j in range(g_lo, g_hi + 1)
                       if j in blocked or _pair_value(measurement, j) is None)
        seg_bounds: list[tuple[int, int]] = []  # (lo_pair, hi_pair) inclusive
        cur_lo = g_lo
        for s in stops:
            if s > cur_lo:
                seg_bounds.append((cur_lo, s - 1))
            cur_lo = s + 1
        if cur_lo <= g_hi:
            seg_bounds.append((cur_lo, g_hi))

        for lo, hi in seg_bounds:
            try:
                start_fn, end_fn = _contiguous_post_interval(pairs, lo, hi)
            except _Segmentation:
                extra_gaps.append({
                    "reason": "decode_misalignment",
                    "frame_range": {"domain": "frame_no",
                                    "start": pairs[lo + 1]["frame_no"],
                                    "end": pairs[hi]["frame_no"] + 1},
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
            # peak-centered sub-runs whose PLATEAU lies inside THIS segment
            merged_runs = []
            for pk in seg_peaks:
                rlo, rhi = runs[peaks.index(pk)]
                mlo, mhi = max(rlo, lo), min(rhi, hi)
                if mlo <= mhi:
                    merged_runs.append([mlo, mhi])
            merged_runs.sort()
            _emit(sh, seg_peaks, lo, hi, start_fn, end_fn)

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
