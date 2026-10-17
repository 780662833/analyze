"""Object-level visual constraint analysis: stable data contracts.

This subpackage defines the two stable artifacts produced by the object-level
analysis stage (``object_tracks.json`` / ``object_constraints.json``). It is
kept intentionally small in v1: the contracts are built and validated here,
while the detector/segmenter/tracker adapters and the orchestration entry
``analyze.py`` are added in later steps.

The contracts encode three architectural invariants (see ``schemas.py``):
time is PTS-authoritative, JSON is index-only, and thresholds/weights are
uncalibrated until a labeled set exists.
"""
from __future__ import annotations

from .schemas import (
    CANONICAL_TIME_SOURCE,
    COORDINATE_SPACE,
    GATE_STATUSES,
    OBJECT_CONSTRAINTS_SCHEMA_VERSION,
    OBJECT_TRACKS_SCHEMA_VERSION,
    build_object_constraints,
    build_object_tracks,
    validate_object_constraints,
    validate_object_tracks,
)
from .io import (
    build_constraints_from_l0,
    build_tracks_from_l0,
    load_l0,
    read_frames,
    resolve_window,
)
from .detector import BBox, Detector, StaticBoxDetector
from .segmenter import BoxMaskSegmenter, Segmenter, StaticBoxSegmenter
from .tracker import NullPointTracker, OpenCVLucasKanadeTracker, PointTracker
from .quality import aggregate_gate, assess_frame, assess_window
from .preview import render_preview, render_preview_from_artifacts
from .analyze import run_analysis, run_frames, run_video
from .video_io import decode_window, video_info

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_TIME_SOURCE",
    "COORDINATE_SPACE",
    "GATE_STATUSES",
    "OBJECT_CONSTRAINTS_SCHEMA_VERSION",
    "OBJECT_TRACKS_SCHEMA_VERSION",
    "build_object_constraints",
    "build_object_tracks",
    "validate_object_constraints",
    "validate_object_tracks",
    "build_constraints_from_l0",
    "build_tracks_from_l0",
    "load_l0",
    "read_frames",
    "resolve_window",
    "BBox",
    "Detector",
    "StaticBoxDetector",
    "Segmenter",
    "StaticBoxSegmenter",
    "BoxMaskSegmenter",
    "NullPointTracker",
    "PointTracker",
    "OpenCVLucasKanadeTracker",
    "aggregate_gate",
    "assess_frame",
    "assess_window",
    "render_preview",
    "composite_local_overlay",
    "verify_local_edit_output",
    "render_preview_from_artifacts",
    "run_analysis",
    "run_frames",
    "run_video",
    "decode_window",
    "video_info",
]