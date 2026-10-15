# Third-party components and license review

This source package does **not** vendor third-party source code or model weights. It calls installed tools/libraries through Python adapters or command-line interfaces. Before redistributing a packaged deployment, generate an SBOM from the exact locked environment and preserve the notices for the versions actually shipped.

| Component | Role | Upstream license / review note |
|---|---|---|
| FFmpeg / FFprobe | Media facts, decoding, `scdet` | FFmpeg build configuration matters: LGPL/GPL status depends on how that binary was compiled. Check `ffmpeg -L` and the distributor's notices. |
| PySceneDetect | Formal detector A | Upstream repository declares BSD 3-Clause: https://github.com/Breakthrough/PySceneDetect/blob/main/LICENSE |
| OpenCV Python package | Evidence frame extraction / fallback | OpenCV 4.5.0+ uses Apache 2.0; older lines differ. Check the actual installed version and package notices: https://github.com/opencv/opencv/wiki/ChangeLog |
| TransNetV2 | Optional detector B | Upstream repository declares MIT: https://github.com/soCzech/TransNetV2/blob/master/LICENSE. This does not automatically clear separate model-weight, dataset, or transitive-dependency questions; review the exact files downloaded. |

Notes:
- This table is a practical engineering checklist, not legal advice.
- The user's own project license has not been assigned by this package.
- This directory includes no TransNetV2 weights and no copied OpenMontage repository implementation.
