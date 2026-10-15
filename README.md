# OpenMontage Intelligence

See [README_快速开始.md](README_快速开始.md) for installation, CLI, outputs, and validation limitations.

The module boundary is explicit:
- `omintel.media_facts`: L0 source identity and canonical PTS frame index.
- `omintel.detectors`: independent L1 detector adapters.
- `omintel.fusion`: L2 one-to-one fusion, provisional timeline, and evidence package.
- `omintel.pipeline`: CLI orchestration and run report.
