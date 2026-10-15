"""Data contracts for the object-level visual constraint layer.

Two artifacts are defined here and are stable across detector/segmenter/
tracker/model replacements:

- ``object_tracks.json``  -- an *index* over per-frame masks, bboxes and
  motion, plus object identity and provenance. It never inlines heavy
  per-frame data; those live in sidecar files referenced by this index.
- ``object_constraints.json`` -- the edit contract consumed by downstream
  compositing/editing. It separates hard constraints (programmatically
  enforceable) from soft constraints (weighted, calibrated later).

Three invariants are baked into the validators below, each fixing a known
architectural issue:

1. **Time is PTS-authoritative.** ``frame_no`` is only a lookup key; the
   authoritative time is ``pts_s`` resolved through the canonical
   ``frames.csv`` from ``omintel.media_facts``. Time is never derived from
   ``frame_no / fps``, because the source may be VFR or have non-monotonic
   PTS.
2. **JSON is index-only.** Masks/bboxes/flow are never embedded as arrays in
   JSON; they are referenced by sidecar file paths.
3. **Thresholds/weights are uncalibrated by default.** A ``null`` weight is
   valid, and the quality gate defaults to ``REVIEW`` until a labeled test
   set exists. No objective correctness number is fabricated.
"""
from __future__ import annotations

from typing import Any

OBJECT_TRACKS_SCHEMA_VERSION = "openmontage.object_tracks.v1"
OBJECT_CONSTRAINTS_SCHEMA_VERSION = "openmontage.object_constraints.v1"

CANONICAL_TIME_SOURCE = "ffprobe_best_effort_timestamp_time"
COORDINATE_SPACE = "coded_pixels_pre_rotation"

GATE_STATUSES = ("PASS", "REPAIR", "REVIEW")

#: Reason attached to REVIEW gates while thresholds/weights remain uncalibrated.
UNcalibrated_REASON = "thresholds and weights uncalibrated; no labeled test set yet"

#: Keys that must be present in the ``tracks`` block and must reference
#: sidecar files (so the JSON stays index-only rather than inlining data).
_INDEX_TRACKS_KEYS = ("bbox_file", "mask_dir", "mask_index_csv", "motion_file")


class ContractError(ValueError):
    """Raised when a contract document violates an architectural invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def build_object_tracks(
    *,
    video_id: str,
    source_sha256: str,
    object_id: str,
    shot_id: str | None,
    coded_size: tuple[int, int],
    display_size: tuple[int, int],
    rotation_deg: int,
    start_frame_no: int,
    start_pts_s: float,
    end_frame_no: int,
    end_pts_s: float,
    likely_variable_frame_rate: bool,
    bbox_file: str,
    mask_dir: str,
    mask_index_csv: str,
    motion_file: str,
    per_frame_quality_csv: str,
    class_name: str | None = None,
    reference_asset_id: str | None = None,
    detector: dict[str, Any] | None = None,
    segmenter: dict[str, Any] | None = None,
    tracker: dict[str, Any] | None = None,
    mask_source: str | None = None,
    mask_precision: str | None = None,
    initialization_file: str | None = None,
) -> dict[str, Any]:
    """Build a canonical ``object_tracks.json`` document.

    ``start_pts_s``/``end_pts_s`` must come from the canonical ``frames.csv``
    PTS timeline, never from ``frame_no / fps``. Unverified model fields
    (``model_version``/``checkpoint``) default to ``None`` to mark them as
    not-yet-traceable rather than falsely reproducible.
    """
    det = detector or {
        "backend": "grounding_dino", "model_version": None,
        "prompt_texts": [class_name] if class_name else [],
        "params": {},
    }
    seg = segmenter or {
        "backend": "sam2_video", "checkpoint": None, "params": {},
    }
    trk = tracker or {
        "backend": "opencv_sparse_lk",
        "params": {"win_size": [15, 15], "max_level": 3},
    }
    return {
        "schema_version": OBJECT_TRACKS_SCHEMA_VERSION,
        "pipeline_version": "openmontage-object-constraints/0.1.0",
        "video_id": video_id,
        "source_sha256": source_sha256,
        "shot_id": shot_id,
        "object": {
            "object_id": object_id,
            "class": class_name,
            "reference_asset_id": reference_asset_id,
            # generic analysis: the target is any user-specified object class
            # (free-form text prompt), not bound to a specific product category
            "class_scope": "generic_any_class",
            "identity_model": "single_initial_prompt",
            "identity_note": (
                "v1 treats the object as a single identity seeded by one "
                "prompt; no cross-occlusion re-identification is performed."
            ),
        },
        "media_reference": {
            "coded_width": coded_size[0],
            "coded_height": coded_size[1],
            "display_width": display_size[0],
            "display_height": display_size[1],
            "rotation_deg": rotation_deg,
            "coordinate_space": COORDINATE_SPACE,
        },
        "time_base": {
            "canonical_time_source": CANONICAL_TIME_SOURCE,
            "likely_variable_frame_rate": bool(likely_variable_frame_rate),
            "start": {"pts_s": start_pts_s, "frame_no": start_frame_no},
            "end": {"pts_s": end_pts_s, "frame_no": end_frame_no},
        },
        "tracks": {
            "bbox_format": "xyxy_normalized",
            "bbox_coordinate_space": "normalized_relative_to_coded_pixels",
            "bbox_file": bbox_file,
            "mask_format": "per_frame_png",
            "mask_coordinate_space": COORDINATE_SPACE,
            "mask_dir": mask_dir,
            "mask_index_csv": mask_index_csv,
            # how masks were produced: a rectangular placeholder derived from
            # the tracked bbox is NOT a real object outline; downstream must
            # treat it as an interface-validation artifact only
            "mask_source": mask_source or "bbox_rectangle_placeholder",
            "mask_precision": mask_precision or "rectangular_approximation",
            "initialization_file": initialization_file,
            "detection_source": det.get("backend"),
            "segmentation_source": seg.get("backend"),
            "motion_source": trk.get("backend"),
            "motion_coordinate_space": COORDINATE_SPACE,
            "motion_file": motion_file,
            "dense_flow_file": None,
        },
        "per_frame_quality_csv": per_frame_quality_csv,
        "provenance": {
            "detector": det,
            "segmenter": seg,
            "tracker": trk,
        },
    }


def build_object_constraints(
    *,
    video_id: str,
    source_sha256: str,
    object_id: str,
    coded_size: tuple[int, int],
    display_size: tuple[int, int],
    rotation_deg: int,
    start_frame_no: int,
    start_pts_s: float,
    end_frame_no: int,
    end_pts_s: float,
    tracks_index: str = "object_tracks.json",
) -> dict[str, Any]:
    """Build a canonical ``object_constraints.json`` edit contract.

    Soft-constraint weights default to ``None`` (uncalibrated) and the
    quality gate defaults to ``REVIEW`` for the same reason. Region
    invariance is stated to be verified on lossless intermediate frames.
    """
    return {
        "schema_version": OBJECT_CONSTRAINTS_SCHEMA_VERSION,
        "video_id": video_id,
        "source_sha256": source_sha256,
        "object_id": object_id,
        "tracks_index": tracks_index,
        "coordinate_space": COORDINATE_SPACE,
        "media": {
            "coded_width": coded_size[0],
            "coded_height": coded_size[1],
            "display_width": display_size[0],
            "display_height": display_size[1],
            "rotation_deg": rotation_deg,
        },
        "edit_window": {
            "start": {"pts_s": start_pts_s, "frame_no": start_frame_no},
            "end": {"pts_s": end_pts_s, "frame_no": end_frame_no},
        },
        "hard_constraints": [
            {"id": "hc_temporal_boundary", "type": "exclusive_range",
             "value": {"start_pts_s": start_pts_s, "end_pts_s": end_pts_s}},
            {"id": "hc_object_scope", "type": "object_id", "value": object_id},
            {"id": "hc_no_cross_cut_track", "type": "flag", "value": True},
            {"id": "hc_preserve_regions", "type": "region_ref",
             "value": "preserve_regions"},
            {"id": "hc_time_reference", "type": "pts_authoritative",
             "value": {"source": CANONICAL_TIME_SOURCE}},
            {"id": "hc_file_integrity", "type": "same_source_sha256",
             "value": source_sha256},
        ],
        "edit_regions": [
            {"region_id": f"{object_id}_mask", "object_id": object_id,
             "mask_source": f"{tracks_index}#tracks.mask_dir"},
        ],
        "preserve_regions": [
            {
                "region_id": "elsewhere",
                "derivation": "complement_of_edit_mask",
                "verify": "pixel_identical_on_lossless_intermediate",
                "note": (
                    "region invariance is checked on lossless intermediate/"
                    "fused frames, not on the final lossy-encoded output."
                ),
            }
        ],
        "soft_constraints": [
            {"id": "sc_motion_consistency", "metric": "flow_agreement",
             "weight": None},
            {"id": "sc_mask_shape", "metric": "mask_area_ratio_change",
             "weight": None},
            {"id": "sc_appearance", "metric": "appearance_similarity",
             "weight": None},
        ],
        "quality_gate": {
            "status": "REVIEW",
            "reason": UNcalibrated_REASON,
        },
    }


def validate_object_tracks(doc: dict[str, Any]) -> None:
    """Validate the architectural invariants of an ``object_tracks`` doc."""
    _require(doc.get("schema_version") == OBJECT_TRACKS_SCHEMA_VERSION,
             f"schema_version must be {OBJECT_TRACKS_SCHEMA_VERSION}")
    _require(doc.get("video_id") and doc.get("source_sha256"),
             "video_id and source_sha256 are required")

    tb = doc.get("time_base") or {}
    _require(tb.get("canonical_time_source") == CANONICAL_TIME_SOURCE,
             f"time_base.canonical_time_source must be {CANONICAL_TIME_SOURCE}")
    start = tb.get("start") or {}
    end = tb.get("end") or {}
    for side in (start, end):
        _require(isinstance(side.get("frame_no"), int),
                 "time_base start/end require an integer frame_no")
        _require(side.get("pts_s") is not None,
                 "time_base start/end require a pts_s value")
    _require(float(start["pts_s"]) <= float(end["pts_s"]),
             "start.pts_s must be <= end.pts_s")

    mr = doc.get("media_reference") or {}
    _require(mr.get("coordinate_space") == COORDINATE_SPACE,
             f"media_reference.coordinate_space must be {COORDINATE_SPACE}")

    tracks = doc.get("tracks") or {}
    for key in _INDEX_TRACKS_KEYS:
        _require(tracks.get(key), f"tracks.{key} must reference a sidecar file/dir")
    _require("frames" not in tracks and "mask_array" not in tracks,
             "tracks must remain index-only; per-frame data lives in sidecar files")
    # a rectangular placeholder mask must never be mistaken for a real object
    # outline: the producing source is an explicit, required declaration
    _require(bool(tracks.get("mask_source")),
             "tracks.mask_source must declare how masks were produced")
    _require(tracks.get("mask_source") != "segmentation_model"
             or tracks.get("mask_precision") == "object_outline",
             "model-produced masks must declare object_outline precision")


def validate_object_constraints(doc: dict[str, Any]) -> None:
    """Validate the architectural invariants of an ``object_constraints`` doc."""
    _require(doc.get("schema_version") == OBJECT_CONSTRAINTS_SCHEMA_VERSION,
             f"schema_version must be {OBJECT_CONSTRAINTS_SCHEMA_VERSION}")
    _require(doc.get("source_sha256") and doc.get("object_id"),
             "source_sha256 and object_id are required")
    _require(doc.get("coordinate_space") == COORDINATE_SPACE,
             f"coordinate_space must be {COORDINATE_SPACE}")

    hard = {h.get("id"): h for h in doc.get("hard_constraints", [])}
    time_ref = hard.get("hc_time_reference") or {}
    _require((time_ref.get("value") or {}).get("source") == CANONICAL_TIME_SOURCE,
             f"hc_time_reference must use {CANONICAL_TIME_SOURCE}")
    integrity = hard.get("hc_file_integrity") or {}
    _require(bool((integrity.get("value") or "")),
             "hc_file_integrity must carry the source sha256")

    gate = doc.get("quality_gate") or {}
    _require(gate.get("status") in GATE_STATUSES,
             f"quality_gate.status must be one of {GATE_STATUSES}")

    for sc in doc.get("soft_constraints", []):
        weight = sc.get("weight")
        _require(weight is None or isinstance(weight, (int, float)),
                 f"soft_constraints.{sc.get('id')}.weight must be null or numeric")