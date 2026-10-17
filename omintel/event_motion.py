"""L1.5 Commit E (E2a): deterministic frame-pair motion measurement sequence.

This module produces a *measurement record only* -- no event candidates, no
thresholds-as-events, no peak semantics.  Peak criteria live in the follow-up
increment (E2b) so reviewers can audit the raw sequence before any judgment
is applied to it.  Same layering as E1 (``event_measurements``) -> E1b
(``event_extractors``).

Metric contract (mean_abs_diff is a picture-change measure, NOT proof of
motion; see module tests which pin this limitation explicitly):

  For adjacent grayscale frames I_{t-1}, I_t (uint8), per frame pair:
    mean_abs_diff        = mean(|I_t - I_{t-1}|)          over all pixels
    max_abs_diff         = max(|I_t - I_{t-1}|)           single-pixel worst case
    changed_pixel_ratio  = mean(|I_t - I_{t-1}| >= pixel_change_threshold)
                           fraction of pixels whose abs delta reaches the
                           threshold (>=: inclusive by definition, recorded
                           in params/provenance)

  The three metrics are complementary dimensions; they are deliberately NOT
  fused into a single "motion confidence" here.

Frozen contract details (per review checklist):
  * Timestamp semantics: frame pair (prev, cur) is stamped with the
    authoritative PTS of ``cur`` -- "the change was observed at the current
    sample".  Never interpolated from fps, never derived from frame_no/fps.
  * Boundary adjacency: boundary frame number b means the cut falls between
    frames b-1 and b (ShotSpan convention: b is the first frame owned by the
    NEXT shot).  Exactly the pair (b-1, b) is flagged
    ``adjacent_to_boundary=True``; no other pair is ever flagged true.
    Flagging is metadata only -- values are kept, not zeroed or dropped.
  * Coverage denominator: coverage_ratio = valid_pairs / theoretical_pairs,
    where theoretical_pairs = len(frames) - 1.  Invalid pairs stay in the
    denominator (as gaps), so coverage can never be inflated by decode
    failures.
  * Provenance: every metric parameter (including the auxiliary pixel
    threshold) is recorded in AlgorithmProvenance.params.  The pixel
    threshold is described as a measurement parameter, not a validated
    cross-material motion standard.
  * No silent zero-fill: an invalid pair has null metrics plus an explicit
    CoverageGap (reason distinguishes missing_pts vs measurement_unavailable);
    static frames legitimately read ~0 and are never confused with gaps.
  * Determinism: floats rounded at one canonical point; identical inputs +
    params + ALGORITHM_VERSION serialize byte-identically.
  * Isolation: reads numpy grayscale arrays provided by the caller; performs
    no decoding, no disk writes, no LLM, no GPU, does not touch L0/L0.5/L1/L2
    outputs or events.json.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

#: bump whenever measurement semantics change; recorded in provenance.
ALGORITHM_VERSION = "1.0.0"

#: name recorded in provenance for downstream consumers.
METRIC_NAME = "frame_pair_abs_diff"

DEFAULT_PARAMS: dict[str, Any] = {
    # round stats to this many decimals so float noise cannot leak into output
    "stat_precision": 6,
    # auxiliary metric parameter: a pixel counts as "changed" when its
    # absolute uint8 delta is >= this value.  Measurement parameter only --
    # applicability across materials is NOT yet validated.
    "pixel_change_threshold": 8,
}


def _rounded(value: float, precision: int) -> float:
    return float(round(float(value), precision))


def compute_motion_sequence(
    frames: list[dict[str, Any]],
    images: list[np.ndarray | None],
    *,
    boundary_frame_nos: list[int] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute per-frame-pair picture-change measurements over the L0 timeline.

    Parameters
    ----------
    frames:
        L0 frames.csv rows (dicts with at least ``frame_no`` and
        ``pts_time_s``; extra columns ignored).  List order defines
        adjacency; consecutive rows must have strictly increasing frame_no
        (otherwise the input is misaligned and we raise).
    images:
        One grayscale uint8 ndarray per frame row, or ``None`` when the
        frame could not be read/decoded.  Length must equal ``len(frames)``.
    boundary_frame_nos:
        Shot-boundary frame numbers b under the ShotSpan convention (b is
        the first frame of the next shot).  The pair (b-1, b) is flagged
        ``adjacent_to_boundary``.  Values near such pairs are still measured
        and reported -- suppression/classification is E2b's job.
    params:
        Overrides merged over DEFAULT_PARAMS.

    Returns
    -------
    dict
        Measurement record::

            {
              "metric_name", "algorithm_version", "params",
              "pair_count", "valid_pair_count", "coverage_ratio",
              "frame_pairs": [
                {"pair_index", "prev_frame_no", "frame_no",
                 "pts_s",                     # cur frame's authoritative PTS
                 "mean_abs_diff", "max_abs_diff", "changed_pixel_ratio",
                 "valid", "adjacent_to_boundary",
                 "invalid_reason"},           # None | "missing_pts" | ...
                ...
              ],
              "gaps": [CoverageGap dicts...],
              "input_digest", "provenance",
            }

    Raises
    ------
    ValueError
        On length mismatch, fewer than two frames, non-increasing frame_no
        sequences, or shape mismatch between a readable adjacent pair.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    precision = int(p["stat_precision"])
    pixel_thr = int(p["pixel_change_threshold"])

    if len(frames) != len(images):
        raise ValueError(
            f"frames/images length mismatch: {len(frames)} vs {len(images)} "
            "(silent misalignment is forbidden)"
        )
    n = len(frames)
    if n < 2:
        raise ValueError("need at least two frames to measure a frame pair")

    frame_nos = [int(fr["frame_no"]) for fr in frames]
    for a, b in zip(frame_nos, frame_nos[1:]):
        if b <= a:
            raise ValueError(
                f"frame_no must strictly increase in list order: {a} -> {b}"
            )

    # Authoritative PTS consumed verbatim from pts_time_s strings; never
    # re-derived from frame_no / fps.
    def _pts(fr: dict[str, Any]) -> float | None:
        raw = fr.get("pts_time_s")
        if raw is None or raw == "":
            return None
        try:
            return _rounded(float(raw), precision)
        except (TypeError, ValueError):
            return None

    pts_list = [_pts(fr) for fr in frames]

    boundary_set = set(int(b) for b in (boundary_frame_nos or []))

    pairs: list[dict[str, Any]] = []
    gap_records: list[tuple[int, str]] = []  # (pair_index, reason)

    for i in range(n - 1):
        prev_fn, cur_fn = frame_nos[i], frame_nos[i + 1]
        cur_pts = pts_list[i + 1]
        prev_img, cur_img = images[i], images[i + 1]

        invalid_reason: str | None = None
        if cur_pts is None:
            invalid_reason = "missing_pts"
        elif prev_img is None or cur_img is None:
            invalid_reason = "measurement_unavailable"

        mean_d = max_d = ratio = None
        if invalid_reason is None:
            a = np.asarray(prev_img, dtype=np.uint8)
            bimg = np.asarray(cur_img, dtype=np.uint8)
            if a.shape != bimg.shape:
                raise ValueError(
                    f"shape mismatch between readable pair frames "
                    f"{prev_fn}({a.shape}) and {cur_fn}({bimg.shape})"
                )
            diff = np.abs(bimg.astype(np.int16) - a.astype(np.int16))
            mean_d = _rounded(float(diff.mean()), precision)
            max_d = _rounded(float(diff.max()), precision)
            ratio = _rounded(
                float((diff >= pixel_thr).mean()), precision
            )

        pairs.append({
            "pair_index": i,
            "prev_frame_no": prev_fn,
            "frame_no": cur_fn,
            # timestamp semantics: change observed at CURRENT frame's PTS
            "pts_s": cur_pts,
            "mean_abs_diff": mean_d,
            "max_abs_diff": max_d,
            "changed_pixel_ratio": ratio,
            "valid": invalid_reason is None,
            # frozen definition: boundary b flags exactly the pair (b-1, b)
            "adjacent_to_boundary": cur_fn in boundary_set,
            "invalid_reason": invalid_reason,
        })
        if invalid_reason is not None:
            gap_records.append((i, invalid_reason))

    valid_count = sum(1 for pr in pairs if pr["valid"])
    theoretical = n - 1
    coverage = _rounded(valid_count / theoretical, precision)

    # Merge contiguous same-reason pairs into half-open [start_fn, end_fn)
    # gap intervals (end = frame AFTER the run's last pair's cur frame is
    # expressed through frame indices on the pair domain).
    gaps: list[dict[str, Any]] = []
    run: list[tuple[int, str]] = []
    def _flush(run_items: list[tuple[int, str]]) -> None:
        if not run_items:
            return
        start_idx = run_items[0][0]
        end_idx = run_items[-1][0]  # inclusive pair index
        gaps.append({
            "reason": run_items[0][1],
            "frame_range": {
                "domain": "frame_no",
                "start": frame_nos[start_idx],       # prev frame of first bad pair
                "end": frame_nos[end_idx + 1] + 1,   # exclusive: covers full span
            },
            "detail": (
                f"{len(run_items)} contiguous frame pair(s) unmeasurable "
                f"(pair indices {start_idx}..{end_idx})"
            ),
        })
    for idx, rec in enumerate(gap_records):
        if run and (rec[0] != run[-1][0] + 1 or rec[1] != run[-1][1]):
            _flush(run)
            run = []
        run.append(rec)
    _flush(run)

    content = {
        "metric_name": METRIC_NAME,
        "algorithm_version": ALGORITHM_VERSION,
        "params": p,
        "frame_nos": frame_nos,
        "pts_list": pts_list,
        "pairs": pairs,
        "gaps": gaps,
        "boundary_frame_nos": sorted(boundary_set),
    }
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    return {
        "metric_name": METRIC_NAME,
        "algorithm_version": ALGORITHM_VERSION,
        "params": dict(p),
        "pair_count": theoretical,
        "valid_pair_count": valid_count,
        "coverage_ratio": coverage,
        "frame_pairs": pairs,
        "gaps": gaps,
        "input_digest": digest,
        "provenance": {
            "name": METRIC_NAME,
            "algorithm_version": ALGORITHM_VERSION,
            "params": dict(p),
            "notes": (
                "mean_abs_diff/max_abs_diff/changed_pixel_ratio are "
                "picture-change measurements; high values may equally come "
                "from hard cuts, flashes or exposure changes. Pixel "
                "threshold is a measurement parameter, not a validated "
                "cross-material motion standard."
            ),
        },
    }
