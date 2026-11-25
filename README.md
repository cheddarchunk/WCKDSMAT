# WCKDSMAT

Webcam-based facial kinematics for speech and oral-motor research.

> **Status: pre-release (0.1.0.dev0).** The stage scripts were consolidated from research code and are being refactored into a library with a CLI. The interfaces will change.

## Pipeline

| Stage | Module | Input → Output |
|---|---|---|
| Landmark extraction | `wckdsmat.extract` | video → MediaPipe Face Mesh landmarks |
| Calibration (Method G) | `wckdsmat.calibrate` | pixel landmarks → mm |
| Preprocessing | `wckdsmat.preprocess` | low-pass, head normalization, mouth residuals |
| Kinematics | `wckdsmat.kinematics_imll`, `kinematics_grouped`, `head` | articulator and head-movement time series |
| Validation | `wckdsmat.validate` | agreement with motion capture (CTX) |
| QC viewer | `viewer/` (in progress) | interactive review and segmentation |

## Install (development)

```bash
pip install -e ".[all,dev]"
```

## Signal conventions

See [docs/fix_log.md](docs/fix_log.md). In short: 3D traces are excursion, error or speed magnitudes, never `||position||`.

## Citation

TODO
