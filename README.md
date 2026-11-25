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

## Configuration

One TOML file controls the whole pipeline: which MediaPipe Face Mesh points to track and what to call them, named groups (head anchors, lips, jaw), derived points (e.g. `IMLL = MLL - mean(LJ, RJ)`), and every stage's settings. The `speech` preset holds the defaults for speech and oral-motor work. Override only what you need:

```bash
wckdsmat config init my_study.toml      # start from the speech preset
wckdsmat config check my_study.toml     # report every problem at once
wckdsmat config show my_study.toml      # print the resolved settings
```

## Signal conventions

See [docs/fix_log.md](docs/fix_log.md). In short: 3D traces are excursion, error or speed magnitudes, never `||position||`.

## Citation

TODO
