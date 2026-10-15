# Self-test results

Date: 2026-10-14 (UTC environment clock)

## Automated tests

Command:

```bash
python -m unittest discover -s tests -v
```

Result: **4 tests passed**.

- L2 one-to-one candidate fusion emits agreement/A-only/B-only correctly.
- L2 matching maximizes the number of valid pairs before minimizing total timestamp delta.
- PTS nearest-frame lookup picks the nearest canonical timestamp.
- End-to-end synthetic pipeline creates L0 metadata, both detector outputs, fused boundaries, provisional shots, event JSON, evidence index, and evidence images.

## Persisted smoke run

Fixture: `tests/synthetic_three_scenes.mp4`, a generated 3-second 320x180, 30-FPS video with hard changes at 1.0s and 2.0s.

Run mode: OpenCV HSV fallback (A) + FFmpeg `scdet` (B). The formal PySceneDetect branch was not run because PySceneDetect is not installed in this execution environment and the package installer could not reach PyPI. The TransNetV2 branch was not run because its official inference package/weights are not installed.

Observed:

- Decoded frames: 90.
- Frames missing PTS: 0.
- Non-monotonic adjacent PTS: 0.
- OpenCV fallback candidates: 2 (1.0s and 2.0s).
- FFmpeg `scdet` candidates: 1 (2.0s).
- Fusion: 1 corroborated candidate (2.0s), 1 A-only candidate (1.0s), 0 B-only candidates.
- Evidence events: 2; both have before/at/after image files and SHA-256 hashes.

This run validates the plumbing and artifact generation only. It is **not** evidence of real-video detection accuracy. The missing 1.0s boundary from FFmpeg also illustrates why detector disagreements must remain visible for review.
