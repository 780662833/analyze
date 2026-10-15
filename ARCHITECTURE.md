# Architecture and module contracts

## Data flow

`source video -> L0 media facts -> L1-A detector + L1-B detector -> L2 candidate fusion -> evidence pack`

The orchestrator is intentionally thin. Each detector emits candidate boundaries with a detector-specific provenance record; the fusion module does not overwrite or reinterpret the original detector output.

## Module 1 — `omintel.media_facts`

**Input:** local video path and output directory.

**Outputs:**
- `media_facts.json`: SHA-256, file size/path/name, tool versions, container and stream metadata, resolution/rotation, duration, frame timing diagnostics.
- `frames.csv`: one row per ffprobe-decoded video frame; the canonical timestamp preference is `best_effort_timestamp_time`, with explicitly recorded fallback fields.

**Invariants:** the source file is not modified; frame timestamps are not inferred solely from `frame_no / fps`; the source hash accompanies downstream artifacts.

## Module 2 — `omintel.detectors`

Two independent detector adapters are exposed:
- A: PySceneDetect `ContentDetector` (preferred formal baseline) or a clearly labeled OpenCV HSV fallback (wiring tests only).
- B: FFmpeg `scdet` (lightweight default) or an optional TransNetV2 inference adapter.

Each detector returns a normalized record containing `boundary_id`, `frame_no`, `time_s`, `detector`, `score` when supplied by the backend, and detector parameters. Raw CSV/metadata is retained separately.

**Do not compare score numbers across detectors.** Score scales, threshold definitions, and backend semantics differ. Candidate timestamps are mapped onto the L0 timeline where possible.

## Module 3 — `omintel.fusion`

**Candidate matching:** monotonic dynamic programming maximizes the number of one-to-one pairs within tolerance, then minimizes total absolute timestamp deviation. Unmatched candidates remain visible as A-only or B-only. This prevents one candidate from being consumed by a locally closer pair if that reduces total corroboration count.

**Outputs:**
- `shot_candidates.csv/json`: flattened candidate table with detector provenance.
- `provisional_shots.csv`: a candidate-based partition for review; not ground truth.
- `events.json`: event-level source references and neighboring evidence frame references.
- `evidence_index.json` and `evidence/<candidate_id>/{before,at,after}.jpg`: saved frames with SHA-256 hashes.

**Evidence semantics:** each event means “shot-boundary candidate,” not “confirmed cut.” Before/at/after frames provide inspectable visual context; they do not prove whether a boundary is a cut, dissolve, camera move, or action change.

## Design choices

- Keep the deterministic signal pipeline separate from LLM interpretation.
- Preserve raw outputs from every detector so threshold tuning can be reproduced.
- Use JSON/CSV in v0.2 so results are inspectable and portable; Parquet and a database can be added when throughput requires them.
- Keep local processing as the default. Cloud APIs should be adapters, not mandatory dependencies.
- Add action/subject events, OCR/ASR alignment, semantic interpretation, and Blueprint generation only after these three modules are validated on a labeled reference set.

## Current non-goals

- No automatic claim of action semantics or creative intent.
- No precision/recall/F1 until a human-reviewed ground-truth dataset exists.
- No claim that OpenCV fallback equals PySceneDetect.
- No claim that `agreement` equals correctness.
